import threading
import queue
import usb1
import logging
from typing import Callable

from PyQa40x.registers import Registers

logger = logging.getLogger(__name__)


class AdcOverflowError(RuntimeError):
    """
    Raised when the ADC receive buffer exceeds its high-water mark.

    The capture pipeline requires the consumer to read captured data
    continuously (see ``consume_adc``).  If the consumer falls behind, the
    receive buffer grows without bound toward OOM.  Rather than leak memory,
    the stream treats this as a hard error: the worker thread latches it and
    ``consume_adc`` re-raises it on the consumer's thread.
    """


class Stream:
    """
    Class to manage buffer streaming, with event worker thread.
    """

    TRANSFER_TIMEOUT_MS = 1000  # Timeout for USB transfers in milliseconds
    TRANSFER_BYTES = 16384  # 16 kB per USB bulk transfer (hardware chunk size)
    # cap on unread ADC data before overflow
    MAX_BUFFERED_BYTES = 256 * 1024 * 1024

    def __init__(
        self,
        context: usb1.USBContext,
        device: usb1.USBDeviceHandle,
        registers: Registers,
    ) -> None:
        """
        Initialize the Stream.

        Args:
            context (usb1.USBContext): The USB context.
            device (usb1.USBDeviceHandle): The USB device handle.
            registers (Registers): The Registers instance for control writes.
        """
        self.context = context
        self.device = device
        self.registers = registers
        self.endpoint_read = usb1.ENDPOINT_IN | 0x02  # EP2 in
        self.endpoint_write = usb1.ENDPOINT_OUT | 0x02  # EP2 out
        self.dacQueue = queue.Queue(
            maxsize=5
        )  # Max. 5 overlapping buffers in flight, block on more
        self.adcQueue = queue.Queue()  # Unlimited queue for received data buffers
        self.transfer_helper = usb1.USBTransferHelper()  # Use the callback dispatcher
        # Handle COMPLETED transfers normally; also handle ERROR/TIMED_OUT/CANCELLED
        # so the queues are always drained and dacQueue.put() never deadlocks.
        for status in (
            usb1.TRANSFER_COMPLETED,
            usb1.TRANSFER_ERROR,
            usb1.TRANSFER_TIMED_OUT,
            usb1.TRANSFER_CANCELLED,
            usb1.TRANSFER_STALL,
            usb1.TRANSFER_NO_DEVICE,
            usb1.TRANSFER_OVERFLOW,
        ):
            self.transfer_helper.setEventCallback(status, self.callback)
        self.thread: threading.Thread | None = None
        self.running = False
        # In-flight transfers, keyed by identity; pruned in callback, bulk-cancelled in stop.
        self._pending_transfers: set = set()
        # raised to break the worker out of stuck queues
        self._force_stop = threading.Event()
        self._transfer_lock = threading.Lock()  # protects _pending_transfers
        self._epoch = 0  # session counter; stale-session callbacks are ignored
        self._adc_data = queue.Queue()  # received ADC data, one chunk per item
        self._adc_data_carry = bytearray()  # leftover ADC bytes between consume_adc calls
        # latched, raised by consume_adc
        self._overflow_error: AdcOverflowError | None = None
        self._max_queued_chunks = self.MAX_BUFFERED_BYTES // self.TRANSFER_BYTES

    @property
    def overflow_error(self) -> "AdcOverflowError | None":
        """Latched ADC overflow error, or None if the buffer never overran."""
        return self._overflow_error

    def start(self) -> None:
        """
        Start streaming and spawn the worker thread.
        """
        self._force_stop.clear()
        self._epoch += 1  # new session; callbacks from prior sessions are ignored
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.running = True
        self.thread.start()
        self._adc_data = queue.Queue()  # Reset the ADC data queue
        self._adc_data_carry = bytearray()  # Reset the carry buffer
        self._overflow_error = None  # Reset the overflow error
        self.registers.write(8, 0x05)  # Start streaming

    def stop(self) -> None:
        """
        Stop streaming and end the worker thread.

        Sets ``running`` to False and cancels in-flight USB transfers so their
        callbacks fire and drain the queues, which allows the worker loop to
        exit naturally.  If the worker has not exited within 5 seconds (e.g.
        because a USB cancellation is slow), ``_force_stop`` is set so the
        worker breaks unconditionally after its current
        ``handleEventsTimeout(0.1)`` call (≤ 100 ms), and we give it one
        further second to join before proceeding.
        """
        # Cancel any in-flight transfers so their callbacks fire and drain the queues,
        # allowing the worker thread to exit its loop. The lock ensure no transfer is added
        # to the list while we are cancelling.
        with self._transfer_lock:
            self.running = False
            for transfer in self._pending_transfers:
                try:
                    transfer.cancel()
                except Exception:  # pylint: disable=broad-except
                    pass
            self._pending_transfers.clear()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                # Worker is stuck (slow USB cancellation).  Force it out so
                # context.close() does not deadlock.
                self._force_stop.set()
                self.thread.join(timeout=1.0)
        self.registers.write(8, 0x00)  # Stop streaming

    def worker(self) -> None:
        """
        Event loop for the asynchronous transfers.

        Runs until *both* conditions are true:
        * ``running`` is False (``stop()`` has been called), and
        * both USB queues are empty (all callbacks have fired).

        An early exit is also triggered by ``_force_stop`` (set by ``stop()``
        when the normal join times out) so that ``context.close()`` is never
        called while this thread is still inside ``handleEventsTimeout``.
        """
        while self.running or not (
            self.dacQueue.empty() and self.adcQueue.empty()
        ):  # Play until the last
            if self._force_stop.is_set():
                break
            # 100 ms timeout prevents blocking forever
            self.context.handleEventsTimeout(0.1)

    def callback(self, transfer: usb1.USBTransfer) -> None:
        """
        Handle a completed transfer (runs on the worker thread).

        On a completed ADC read, buffer the bytes; if the receive queue passes
        its limit, latch an AdcOverflowError and stop so consume_adc can raise it.

        Args:
            transfer (usb1.USBTransfer): The USB transfer that has completed.
        """
        current = transfer.getUserData() == self._epoch  # ignore stale-session data
        if transfer.getEndpoint() == self.endpoint_read:
            if current and transfer.getStatus() == usb1.TRANSFER_COMPLETED:
                if self._overflow_error is None:
                    # Copy: the transfer buffer is freed once we prune the transfer.
                    self._adc_data.put(bytes(transfer.getBuffer()))
                if self._adc_data.qsize() > self._max_queued_chunks:
                    self._overflow_error = AdcOverflowError(
                        "ADC receive queue exceeded high-water mark "
                        f"({self._adc_data.qsize()} chunks); consumer not "
                        "reading capture fast enough."
                    )
                    self.running = False  # unwind feeder + worker
                    logger.error(str(self._overflow_error))
            self.adcQueue.get_nowait()  # Always drain the queue
        else:
            self.dacQueue.get_nowait()  # Always drain the queue

        # Prune this finished transfer so _pending_transfers stays bounded during
        # a long capture; discard() is a no-op if stop() already cleared it.
        with self._transfer_lock:
            self._pending_transfers.discard(transfer)

    def write(self, buffer: bytes) -> None:
        """Submit one DAC playback buffer and one matching ADC read transfer."""
        with self._transfer_lock:
            if not self.running:
                raise RuntimeError(
                    "Stream is not running; call start() first.")
            dac = self.device.getTransfer()
            dac.setBulk(self.endpoint_write, buffer, self.transfer_helper,
                        self._epoch, self.TRANSFER_TIMEOUT_MS)
            dac.submit()
            self._pending_transfers.add(dac)
            read = self.device.getTransfer()
            read.setBulk(self.endpoint_read, len(buffer), self.transfer_helper,
                         self._epoch, self.TRANSFER_TIMEOUT_MS)
            read.submit()
            self._pending_transfers.add(read)
        self.dacQueue.put(dac)
        self.adcQueue.put(read)

    def write_zeros(self, chunk_bytes: int = TRANSFER_BYTES) -> None:
        """Feed the ADC pipeline with a zero DAC buffer (continuous capture)."""
        self.write(bytes(chunk_bytes))

    def consume_adc(
        self,
        need_bytes: int,
        should_keep_waiting: Callable[[], bool] | None = None,
        poll_s: float = 0.005,
    ) -> bytearray:
        """
        Block until `need_bytes` of ADC data are available, then remove and
        return exactly that many bytes from the front of the receive buffer.

        Args:
            need_bytes (int): Number of bytes to consume.
            should_keep_waiting (Callable[[], bool] | None): Polled each iteration;
                when it returns False, stop waiting and return what is available.
            poll_s (float): Poll interval in seconds.

        Returns:
            bytearray: Up to `need_bytes` of ADC data (fewer only on early stop).

        Raises:
            AdcOverflowError: If the receive buffer exceeded its limit because
                captured data was not read fast enough.
        """
        while len(self._adc_data_carry) < need_bytes:
            if self._overflow_error is not None:
                raise self._overflow_error
            if should_keep_waiting is not None and not should_keep_waiting():
                break
            if not self.running and self._adc_data.empty():
                break  # stream stopped and no more data will arrive, return what we have
            try:
                chunk = self._adc_data.get(timeout=poll_s)
                self._adc_data_carry.extend(chunk)
            except queue.Empty:
                continue  # No new data yet, loop and check conditions again

        if self._overflow_error is not None:
            raise self._overflow_error
        take_bytes_count = min(need_bytes, len(self._adc_data_carry))
        out_bytes = self._adc_data_carry[:take_bytes_count]
        del self._adc_data_carry[:take_bytes_count]
        return out_bytes

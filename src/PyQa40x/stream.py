import threading
import queue
import usb1
import logging
from typing import Callable

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

    def __init__(self, context, device, registers):
        """
        Initializes the Stream class.

        Args:
            context (usb1.USBContext): The USB context.
            device (usb1.USBDeviceHandle): The USB device handle.
            registers (Registers): An instance of the Registers class.
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
        # self.received_data = bytearray()  # Collection of received data bytes
        self.thread = None
        self.running = False
        self._pending_transfers = []  # track all submitted transfers for cancellation
        self._force_stop = (
            threading.Event()
        )  # raised to break the worker out of stuck queues
        self._transfer_lock = threading.Lock()  # protects _pending_transfers
        self._adc_data = queue.Queue()  # Queue for received ADC data in chunks
        self._adc_data_carry = (
            bytearray()
        )  # Buffer to carry ADC data not divided into chunks
        self._overflow_error = None  # latched AdcOverflowError, raised by consume_adc
        self._chunks_limit = 16384  # ~256 MB at 16 KB/chunk

    def start(self):
        """
        Starts the streaming and spawns the worker thread.
        """
        self._force_stop.clear()
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.running = True
        self.thread.start()
        self._adc_data = queue.Queue()  # Reset the ADC data queue
        self._adc_data_carry = bytearray()  # Reset the carry buffer
        self._overflow_error = None  # Reset the overflow error
        self.registers.write(8, 0x05)  # Start streaming

    def stop(self):
        """
        Stops the streaming and ends the worker thread.

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

    def write(self, buffer):
        """
        Adds a buffer to the playback queue.

        Args:
            buffer (bytes): The data buffer to be written.
        """
        transfer = self.device.getTransfer()
        transfer.setBulk(
            self.endpoint_write,
            buffer,
            self.transfer_helper,
            None,
            self.TRANSFER_TIMEOUT_MS,
        )
        transfer.submit()  # Asynchronous transfer
        self._pending_transfers.append(transfer)
        self.dacQueue.put(
            transfer
        )  # It doesn't matter what we put in here, queue is just to track the number of pending transfers

        # Submit a USB bulk transfer to read
        read_transfer = self.device.getTransfer()
        read_transfer.setBulk(
            self.endpoint_read,
            16384,
            self.transfer_helper,
            None,
            self.TRANSFER_TIMEOUT_MS,
        )
        read_transfer.submit()  # Asynchronous transfer
        self._pending_transfers.append(read_transfer)
        self.adcQueue.put(
            read_transfer
        )  # It doesn't matter what we put in here, queue is just to track the number of pending transfers

    def worker(self):
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
            self.context.handleEventsTimeout(
                0.1
            )  # 100 ms timeout prevents blocking forever

    def callback(self, transfer):
        """
        Callback of the worker thread to handle completed transfers.

        Args:
            transfer (usb1.USBTransfer): The USB transfer that has completed.
        """
        if transfer.getEndpoint() == self.endpoint_read:
            if transfer.getStatus() == usb1.TRANSFER_COMPLETED:
                if self._overflow_error is None:
                    self._adc_data.put(transfer.getBuffer())
                if self._adc_data.qsize() > self._chunks_limit:
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

    def write_zeros(self, chunk_bytes: int = 16384):
        """
        Sends a zero-filled DAC buffer of `chunk_bytes` bytes and queues one
        matching ADC read transfer.  Functionally identical to write() but
        generates the zero payload internally so no DAC Wave is needed.

        Args:
            chunk_bytes (int): Number of bytes per transfer (must be a multiple
                of 4).  Defaults to 16384 (16 kB) to match the hardware chunk
                size used by write().
        """
        zeros = bytes(chunk_bytes)

        with self._transfer_lock:
            if not self.running:
                raise RuntimeError("Stream is not running; call start() first.")

            dac_transfer = self.device.getTransfer()
            dac_transfer.setBulk(
                self.endpoint_write,
                zeros,
                self.transfer_helper,
                None,
                self.TRANSFER_TIMEOUT_MS,
            )
            dac_transfer.submit()
            self._pending_transfers.append(dac_transfer)
            self.dacQueue.put(dac_transfer)

            read_transfer = self.device.getTransfer()
            read_transfer.setBulk(
                self.endpoint_read,
                chunk_bytes,
                self.transfer_helper,
                None,
                self.TRANSFER_TIMEOUT_MS,
            )
            read_transfer.submit()
            self._pending_transfers.append(read_transfer)
            self.adcQueue.put(read_transfer)

    def consume_adc(
        self,
        need_bytes: int,
        wait_for_needed: Callable[[], bool] | None = None,
        poll_s: float = 0.005,
    ) -> bytearray:
        """
        Blocks until at least `need_bytes` of ADC data are available, then removes
        and returns exactly that many bytes from the front of the receive buffer.


        Args:
            need_bytes (int): Number of bytes to consume.
            wait_for_needed (bool): Optional flag; when False, stop waiting and return what is available.
            poll_s (float): Poll interval in seconds.

        Returns:
            bytearray: Up to `need_bytes` of ADC data (fewer only on early stop).
        """
        while len(self._adc_data_carry) < need_bytes:
            if self._overflow_error is not None:
                raise self._overflow_error
            if wait_for_needed is not None and not wait_for_needed():
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

import threading
import queue
import time
import usb1
import numpy as np


class Stream:
    """
    Class to manage buffer streaming, with event worker thread.
    """

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

    def start(self):
        """
        Starts the streaming and spawns the worker thread.
        """
        self._force_stop.clear()
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.running = True
        self.thread.start()
        self.received_data = bytearray()
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
        self.running = False
        # Cancel any in-flight transfers so their callbacks fire and drain the queues,
        # allowing the worker thread to exit its loop.
        for t in self._pending_transfers:
            try:
                t.cancel()
            except Exception:
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
        transfer.setBulk(self.endpoint_write, buffer, self.transfer_helper, None, 1000)
        transfer.submit()  # Asynchronous transfer
        self._pending_transfers.append(transfer)
        self.dacQueue.put(transfer)  # It doesn't matter what we put in here

        # Submit a USB bulk transfer to read
        read_transfer = self.device.getTransfer()
        read_transfer.setBulk(
            self.endpoint_read, 16384, self.transfer_helper, None, 1000
        )
        read_transfer.submit()  # Asynchronous transfer
        self._pending_transfers.append(read_transfer)
        self.adcQueue.put(read_transfer)  # It doesn't matter what we put in here

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
                self.received_data.extend(
                    transfer.getBuffer()
                )  # Collect received data bytes
            self.adcQueue.get_nowait()  # Always drain the queue
        else:
            self.dacQueue.get_nowait()  # Always drain the queue

    def collect_remaining_adc_data(self):
        """
        Waits for all remaining ADC transfers to complete and returns the collected data.

        Returns:
            bytearray: The collected ADC data.
        """
        while not self.adcQueue.empty():
            time.sleep(0.01)

        return self.received_data

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

        dac_transfer = self.device.getTransfer()
        dac_transfer.setBulk(
            self.endpoint_write, zeros, self.transfer_helper, None, 1000
        )
        dac_transfer.submit()
        self._pending_transfers.append(dac_transfer)
        self.dacQueue.put(dac_transfer)

        read_transfer = self.device.getTransfer()
        read_transfer.setBulk(
            self.endpoint_read, chunk_bytes, self.transfer_helper, None, 1000
        )
        read_transfer.submit()
        self._pending_transfers.append(read_transfer)
        self.adcQueue.put(read_transfer)

    def collect_adc_exact(self, n_bytes: int) -> bytearray:
        """
        Blocks until at least `n_bytes` of ADC data have been collected, then
        returns exactly `n_bytes` (any excess is discarded).  The stream must
        have been started with start() and all required write_zeros() calls
        must have been submitted before calling this.

        Args:
            n_bytes (int): Exact number of bytes to return.

        Returns:
            bytearray: Exactly n_bytes of calibrated ADC data.
        """
        while len(self.received_data) < n_bytes:
            time.sleep(0.01)
        return bytearray(self.received_data[:n_bytes])

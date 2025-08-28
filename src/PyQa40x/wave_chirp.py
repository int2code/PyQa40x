import numpy as np
from typing import Tuple
from PyQa40x.analyzer_params import AnalyzerParams
from PyQa40x.math_general import normalize_spectrum
from PyQa40x.wave import Wave
from PyQa40x.helpers import dbv_to_vpk, dbfs_to_dbv
import PyQa40x.math_chirp as mc
import PyQa40x.math_general as mg
from scipy.fft import rfft, rfftfreq


class WaveChirp(Wave):
    def __init__(self, params: AnalyzerParams):
        # Initialize the parent class with the AnalyzerParams instance
        super().__init__(params)
        self.chirp_buf = None
        self.inv_filter = None
        self.ref_chirp_buf = None
        self.ref_inv_filter = None
        self.ref_freq = None
        self.ref_fft = None

    def gen_chirp_dbv(
        self,
        dbv: float,
        chirp_start_freq: float = 20,
        chirp_stop_freq: float = 20000,
        chirp_width: float = 0.6
    ) -> "WaveChirp":
        """
        Generates a chirp signal with a specified amplitude in dBV.
        """
        vpk: float = dbv_to_vpk(dbv)

        self.chirp_buf, self.inv_filter = mc.chirp_vp(
            self.params.fft_size,
            self.params.sample_rate,
            vpk,
            chirp_start_freq,
            chirp_stop_freq,
            chirp_width,
        )

        # Grab a reference version for 0 dBV = 1.41Vp
        self.ref_chirp_buf, self.ref_inv_filter = mc.chirp_vp(
            self.params.fft_size,
            self.params.sample_rate,
            np.sqrt(2),
            chirp_start_freq,
            chirp_stop_freq,
            chirp_width,
        )
        self.ref_freq, self.ref_fft, _, _ = mc.normalize_and_compute_fft(
            self.ref_chirp_buf, self.ref_inv_filter, self.params.sample_rate
        )

        self.buffer = np.concatenate(
            (np.zeros(self.params.pre_buf), self.chirp_buf, np.zeros(self.params.post_buf))
        )

        return self

    def gen_chirp_dbfs(
        self,
        dbfs: float,
        chirp_start_freq: float = 20,
        chirp_stop_freq: float = 20000,
        chirp_width: float = 0.6,
    ) -> "WaveChirp":
        """
        Generates a chirp signal with a specified amplitude in dBFS.
        """
        dbv = dbfs_to_dbv(dbfs)
        return self.gen_chirp_dbv(dbv, chirp_start_freq, chirp_stop_freq, chirp_width)

    def get_buffer_and_invfilter(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the buffer and the inverse filter.
        """
        return self.buffer, self.inv_filter

    def compute_fft_db(
        self,
        chirp: np.ndarray = None,
        inverse_filter: np.ndarray = None,
        apply_window: bool = False,
        window_start_time: float = 0.001,
        window_end_time: float = 0.005,
        ramp_up_time: float = 0.001,
        ramp_down_time: float = 0.001,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the impulse response and the FFT of a DUT chirp signal.
        """
        if chirp is None:
            chirp = self.chirp_buf
        if inverse_filter is None:
            inverse_filter = self.inv_filter

        _, correction_db = mg.normalize_spectrum(self.ref_freq, self.ref_fft, 1000)

        freq, fft, ir, window = mc.normalize_and_compute_fft(
            chirp,
            inverse_filter,
            self.params.sample_rate,
            apply_window,
            window_start_time,
            window_end_time,
            ramp_up_time,
            ramp_down_time,
        )

        fft = fft + correction_db

        return freq, fft, ir, window

    def compute_rt(self, ir, decay_db):
        """
        Calculates the reverberation time (RT) from the impulse response.
        """
        squared_ir = ir**2
        edc = np.cumsum(squared_ir[::-1])[::-1]
        edc_db = 10 * np.log10(edc / np.max(edc))

        start_idx = np.where(edc_db <= -5)[0][0]
        end_idx = np.where(edc_db <= -5 - decay_db)[0][0]

        slope, intercept = np.polyfit(
            np.arange(start_idx, end_idx) / self.params.sample_rate,
            edc_db[start_idx:end_idx],
            1,
        )

        rt = -(decay_db + 5) / slope
        return rt, edc_db, start_idx, end_idx

    def compute_rel_mag_phase(
        self,
        wave_ref: "WaveChirp",
        buffer_dut: np.ndarray,
        buffer_ref: np.ndarray,
        apply_window: bool = False,
        regularizer_db: float = -300.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute relative magnitude/phase (self / REF) given the raw captured buffers.
        """
        # Impulse responses
        _, _, ir_dut, _ = self.compute_fft_db(buffer_dut, apply_window=apply_window)
        _, _, ir_ref, _ = wave_ref.compute_fft_db(buffer_ref, apply_window=apply_window)

        # FFTs
        H_dut = rfft(ir_dut)
        H_ref = rfft(ir_ref)

        # Stabilize denominator
        eps = 10.0 ** (regularizer_db / 20.0)
        denom = H_ref.copy()
        small = np.abs(denom) < eps
        if np.any(small):
            denom[small] = eps * np.exp(1j * np.angle(denom[small]))

        H_rel = H_dut / denom

        n = len(ir_ref)
        freq = rfftfreq(n, 1.0 / self.params.sample_rate)

        mag_rel = np.abs(H_rel)
        mag_rel = np.maximum(mag_rel, np.finfo(float).tiny)
        mag_rel_db = 20.0 * np.log10(mag_rel)

        phase_rad = np.unwrap(np.angle(H_rel))
        phase_rel_deg = np.degrees(phase_rad)

        return freq, mag_rel_db, phase_rel_deg

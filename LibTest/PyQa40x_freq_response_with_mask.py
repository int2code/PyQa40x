import os
import numpy as np
from typing import Tuple
from scipy.interpolate import interp1d  # needed by verify_trace_within_mask
from PyQa40x.analyzer import Analyzer
from PyQa40x.wave_chirp import WaveChirp
from PyQa40x.series_plotter import SeriesPlotter

# Python code used to play a chirp from the QA40x hardware. The left channel DAC will connect to the
# DUT (such as a low-pass filter). The right channel DAC will connect to the right channel ADC. And the
# left channel ADC will connect to the DUT output. This allows the relative magnitude and phase of the
# DUT to be measured, treating the right channel as the reference. This allows very precise phase measurements
# and because the measurement is relative, the contributions of the QA40x hardware at the band edges are removed.


def load_mask_file(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads a mask file with lines: freq max min
    Returns: freq_mask, max_mask, min_mask as numpy arrays
    """
    freq_mask, max_mask, min_mask = [], [], []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                freq, max_v, min_v = map(float, parts)
                freq_mask.append(freq)
                max_mask.append(max_v)
                min_mask.append(min_v)
    return np.array(freq_mask), np.array(max_mask), np.array(min_mask)


def verify_trace_within_mask(
    freq_trace: np.ndarray,
    value_trace: np.ndarray,
    freq_mask: np.ndarray,
    max_mask: np.ndarray,
    min_mask: np.ndarray
) -> bool:
    """
    Checks if value_trace is within mask at each freq in freq_mask.
    Returns: True if all within mask, else False
    """
    max_interp = interp1d(freq_mask, max_mask, bounds_error=False, fill_value="extrapolate")
    min_interp = interp1d(freq_mask, min_mask, bounds_error=False, fill_value="extrapolate")
    max_vals = max_interp(freq_trace)
    min_vals = min_interp(freq_trace)
    within = np.logical_and(value_trace <= max_vals, value_trace >= min_vals)
    return np.all(within)


# ---------- Main ----------

print("starting test")

analyzer = Analyzer()
params = analyzer.init(
    sample_rate=48000,
    max_input_level=6,
    max_output_level=8,
    fft_size=2**16,
    window_type="flattop"
)

# Dump key parameters
print(params)

# Create chirps (both channels)
wave_dac_left = WaveChirp(params).gen_chirp_dbv(0)
wave_dac_right = WaveChirp(params).gen_chirp_dbv(0)

# Send the DAC buffers to the hardware, and collect the ADC buffers
wave_adc_left, wave_adc_right = analyzer.send_receive(wave_dac_left, wave_dac_right)

# Per-channel FR and IR (no window)
freq, fft_left_db, ir_left, _ = wave_dac_left.compute_fft_db(
    wave_adc_left.get_main_buffer(), apply_window=False
)
_, fft_right_db, ir_right, _ = wave_dac_right.compute_fft_db(
    wave_adc_right.get_main_buffer(), apply_window=False
)

# Relative magnitude/phase (Left vs Right)
freq_rel, mag_rel_db, phase_rel_deg = wave_dac_left.compute_rel_mag_phase(
    wave_dac_right,
    wave_adc_left.get_main_buffer(),
    wave_adc_right.get_main_buffer(),
    apply_window=False
)

# --- Load magnitude mask (optional) ---
mask_file = os.path.join(os.path.dirname(__file__), "mask_freq.txt")
freq_mask, max_mask, min_mask = None, None, None
max_vals, min_vals = None, None
if os.path.exists(mask_file):
    freq_mask, max_mask, min_mask = load_mask_file(mask_file)

    # Interpolate masks onto measured frequency grid
    max_interp = interp1d(freq_mask, max_mask, bounds_error=False, fill_value="extrapolate")
    min_interp = interp1d(freq_mask, min_mask, bounds_error=False, fill_value="extrapolate")
    max_vals = max_interp(freq_rel)
    min_vals = min_interp(freq_rel)

    if verify_trace_within_mask(freq_rel, mag_rel_db, freq_mask, max_mask, min_mask):
        print("mag_rel_db trace is within mask.")
    else:
        print("mag_rel_db trace is OUTSIDE mask!")
else:
    print(f"Mask file not found: {mask_file} (skipping mag mask overlay)")

# --- Load phase mask (optional) ---
mask_phase_file = os.path.join(os.path.dirname(__file__), "mask_phase.txt")
freq_phase_mask, max_phase_mask, min_phase_mask = None, None, None
max_phase_vals, min_phase_vals = None, None
if os.path.exists(mask_phase_file):
    freq_phase_mask, max_phase_mask, min_phase_mask = load_mask_file(mask_phase_file)

    # Interpolate masks onto measured frequency grid
    max_phase_interp = interp1d(freq_phase_mask, max_phase_mask, bounds_error=False, fill_value="extrapolate")
    min_phase_interp = interp1d(freq_phase_mask, min_phase_mask, bounds_error=False, fill_value="extrapolate")
    max_phase_vals = max_phase_interp(freq_rel)
    min_phase_vals = min_phase_interp(freq_rel)

    if verify_trace_within_mask(freq_rel, phase_rel_deg, freq_phase_mask, max_phase_mask, min_phase_mask):
        print("phase_rel_deg trace is within mask.")
    else:
        print("phase_rel_deg trace is OUTSIDE mask!")
else:
    print(f"Mask file not found: {mask_phase_file} (skipping phase mask overlay)")

# ---- Plots ----
tsp = SeriesPlotter(num_columns=2, row_height=2.5)

# DAC time-domain
tsp.add_time_series(wave_dac_left.get_buffer(), "DAC Left")
tsp.add_time_series(wave_dac_right.get_buffer(), "DAC Right")

tsp.newrow()
# ADC time-domain
tsp.add_time_series(wave_adc_left.get_buffer(), "ADC Left")
tsp.add_time_series(wave_adc_right.get_buffer(), "ADC Right")

tsp.newrow()
# Impulse responses (raw, no window)
tsp.add_time_series(ir_left, "IR Left")
tsp.add_time_series(ir_right, "IR Right")

tsp.newrow()
# Individual FR (absolute, dB)
tsp.add_freq_series(freq, fft_left_db,  "FR Left (dB)",  logx=True, units="dB", ymax=20, ymin=-20)
tsp.add_freq_series(freq, fft_right_db, "FR Right (dB)", logx=True, units="dB", ymax=20, ymin=-20)

tsp.newrow()
# Relative Mag plot (overlay mag mask if present)
if max_vals is not None and min_vals is not None:
    tsp.add_freq_series(
        freq_rel,
        mag_rel_db,
        "Relative Mag (Left/Right) [dB]",
        magnitudes_right={
            "Mag Mask Max": max_vals,
            "Mag Mask Min": min_vals
        },
        units="dB",
        units_right="dB",
        ymin=-20, ymax=20,
        ymin_right=-20, ymax_right=20,
        logx=True
    )
else:
    tsp.add_freq_series(freq_rel, mag_rel_db, "Relative Mag (Left/Right) [dB]",
                        logx=True, units="dB", ymax=20, ymin=-20)

# Relative Phase plot (overlay phase mask if present)
if max_phase_vals is not None and min_phase_vals is not None:
    tsp.add_freq_series(
        freq_rel,
        phase_rel_deg,
        "Relative Phase (Left/Right) [deg]",
        magnitudes_right={
            "Phase Mask Max": max_phase_vals,
            "Phase Mask Min": min_phase_vals
        },
        units="deg",
        units_right="deg",
        ymin=-90, ymax=10,
        ymin_right=-90, ymax_right=10,
        logx=True
    )
else:
    tsp.add_freq_series(freq_rel, phase_rel_deg, "Relative Phase (Left/Right) [deg]",
                        logx=True, units="deg", ymax=10, ymin=-90)

tsp.plot()

analyzer.cleanup()
print("done")

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from typing import Union, Dict

class FreqSeriesPlotter:
    def __init__(self, params, num_columns=2, main_title="", main_title_fontsize=16,
                 log_x_axis=True, units='dBV'):
        self.params = params
        self.num_columns = num_columns
        self.main_title = main_title
        self.main_title_fontsize = main_title_fontsize
        self.log_x_axis = log_x_axis
        self.units = units
        self.freq_series_traces = []

    def add_freq_series(self, signal: Union[np.ndarray, Dict[str, np.ndarray]], label: str):
        """
        Add a frequency series.
        - If signal is a single np.ndarray, behaves as before.
        - If signal is a dict[str, np.ndarray], multiple traces will be plotted
          on the same subplot with the dict keys as labels.
        """
        self.freq_series_traces.append({'signal': signal, 'label': label})

    def plot(self):
        num_rows = (len(self.freq_series_traces) + self.num_columns - 1) // self.num_columns
        fig, axs = plt.subplots(num_rows, self.num_columns, figsize=(14, num_rows * 3.5))

        if self.main_title:
            fig.suptitle(self.main_title, fontsize=self.main_title_fontsize)

        # flatten axs for indexing, even if 1x1
        if num_rows == 1 and self.num_columns == 1:
            axs = np.array([axs])
        axs = axs.flat

        for i, trace in enumerate(self.freq_series_traces):
            ax = axs[i]
            signal = trace['signal']
            label = trace['label']

            # helper to convert magnitude array into dB units
            def to_log(mag: np.ndarray) -> np.ndarray:
                if self.units == 'dBV':
                    return 20 * np.log10(np.maximum(mag, np.finfo(float).tiny))
                elif self.units == 'dBu':
                    return 20 * np.log10(np.maximum(mag, np.finfo(float).tiny)) + 2.2
                else:
                    raise ValueError("Unsupported units. Use 'dBV' or 'dBu'.")

            # x-axis freqs (assume all signals same length)
            if isinstance(signal, dict):
                first_array = next(iter(signal.values()))
                fft_freqs = np.fft.rfftfreq(len(first_array) * 2 - 1, 1 / self.params.sample_rate)
                for sublabel, arr in signal.items():
                    ax.plot(fft_freqs, to_log(arr), label=sublabel)
                ax.legend()
                ax.set_title(label)
            else:
                fft_freqs = np.fft.rfftfreq(len(signal) * 2 - 1, 1 / self.params.sample_rate)
                ax.plot(fft_freqs, to_log(signal))
                ax.set_title(label)

            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(f"Amplitude ({self.units})")
            ax.grid(True)
            ax.set_ylim(-150, 10)
            if self.log_x_axis:
                ax.set_xscale('log')
                ax.set_xlim(20, 20000)
                ax.set_xticks([20, 100, 1000, 10000])
                ax.set_xticklabels(['20', '100', '1k', '10k'])

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        rect = patches.Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                 linewidth=1, edgecolor='black', facecolor='none')
        fig.patches.append(rect)

        plt.show()

import numpy as np
import matplotlib.pyplot as plt

class SeriesPlotter:
    def __init__(self, num_columns: int = 2, main_title: str = "",
                 main_title_fontsize: int = 16, row_height: float = 4.0):
        """
        Initializes the SeriesPlotter class.

        Args:
            num_columns (int): Number of columns per row in the plot grid.
            main_title (str): Main title of the plot.
            main_title_fontsize (int): Font size of the main title.
            row_height (float): Height (inches) per row of subplots. Default=4.0.
        """
        self.num_columns: int = num_columns
        self.rows: list[list[dict]] = [[]]
        self.main_title: str = main_title
        self.main_title_fontsize: int = main_title_fontsize
        self.row_height: float = row_height

    def add_time_series(self, signal: np.ndarray, label: str, signal_right: np.ndarray | None = None, num_samples: int = 0,
                        units: str = "Volts", units_right: str = "Volts", ymin: float | None = None, ymax: float | None = None,
                        ymin_right: float | None = None, ymax_right: float | None = None, xmin: float | None = None,
                        xmax: float | None = None, logx: bool = False):
        """Adds a time series plot to the current row."""
        self.rows[-1].append({
            'type': 'time',
            'signal': signal,
            'signal_right': signal_right,
            'label': label,
            'num_samples': num_samples,
            'units': units,
            'units_right': units_right,
            'ymin': ymin,
            'ymax': ymax,
            'ymin_right': ymin_right,
            'ymax_right': ymax_right,
            'xmin': xmin,
            'xmax': xmax,
            'logx': logx
        })

    def add_freq_series(self, freqs: np.ndarray, magnitudes: np.ndarray, label: str,
                        magnitudes_right: np.ndarray | dict[str, np.ndarray] | None = None,
                        num_samples: int = 0, units: str = "dBV", units_right: str = "dBV",
                        ymin: float | None = None, ymax: float | None = None,
                        ymin_right: float | None = None, ymax_right: float | None = None,
                        xmin: float | None = None, xmax: float | None = None,
                        logx: bool = False):
        """Adds a frequency series plot to the current row."""
        if logx:
            xmin = xmin if xmin is not None else 20
            xmax = xmax if xmax is not None else 20000

        self.rows[-1].append({
            'type': 'freq',
            'freqs': freqs,
            'magnitudes': magnitudes,
            'magnitudes_right': magnitudes_right,
            'label': label,
            'num_samples': num_samples,
            'units': units,
            'units_right': units_right,
            'ymin': ymin,
            'ymax': ymax,
            'ymin_right': ymin_right,
            'ymax_right': ymax_right,
            'xmin': xmin,
            'xmax': xmax,
            'logx': logx
        })

    def newrow(self):
        """Starts a new row for the plots."""
        if self.rows[-1]:
            self.rows.append([])

    def plot(self, block: bool = True):
        """Plots the added time and frequency series."""
        mosaic_layout: list[list[str | None]] = []
        for row in self.rows:
            if not row:
                continue
            while len(row) > self.num_columns:
                mosaic_layout.append([trace['label'] for trace in row[:self.num_columns]])
                row = row[self.num_columns:]
            num_elements = len(row)
            if num_elements < self.num_columns:
                span_each = self.num_columns // num_elements
                remainder = self.num_columns % num_elements
                new_row: list[str | None] = []
                for i in range(num_elements):
                    span = span_each + (1 if i < remainder else 0)
                    new_row.extend([row[i]['label']] * span)
                mosaic_layout.append(new_row)
            else:
                mosaic_layout.append([trace['label'] for trace in row])

        label_to_trace: dict[str, dict] = {trace['label']: trace for row in self.rows for trace in row}
        fig, axd = plt.subplot_mosaic(
            mosaic_layout,
            figsize=(5 * self.num_columns, self.row_height * len(mosaic_layout))
        )

        for label, ax in axd.items():
            if label is None:
                continue
            trace = label_to_trace[label]

            if trace['type'] == 'time':
                signal = trace['signal']
                signal_right = trace['signal_right']
                num_samples = trace['num_samples']
                units = trace['units']
                units_right = trace['units_right']
                ymin, ymax = trace['ymin'], trace['ymax']
                ymin_right, ymax_right = trace['ymin_right'], trace['ymax_right']
                xmin, xmax, logx = trace['xmin'], trace['xmax'], trace['logx']

                if num_samples > 0:
                    signal = signal[:num_samples]
                    if signal_right is not None:
                        signal_right = signal_right[:num_samples]

                ax.plot(signal, label=label)
                ax.set_title(label)
                ax.set_xlabel('Sample Index')
                ax.set_ylabel(f'Amplitude ({units})')
                if ymin is not None and ymax is not None:
                    ax.set_ylim(ymin, ymax)
                if xmin is not None and xmax is not None:
                    ax.set_xlim(xmin, xmax)
                if logx:
                    ax.set_xscale('log')

                if signal_right is not None:
                    ax_right = ax.twinx()
                    ax_right.plot(signal_right, 'r', label=f'{label} Right')
                    ax_right.set_ylabel(f'Amplitude ({units_right})', color='r')
                    if ymin_right is not None and ymax_right is not None:
                        ax_right.set_ylim(ymin_right, ymax_right)

            elif trace['type'] == 'freq':
                freqs = trace['freqs']
                num_samples = trace['num_samples']
                magnitudes = trace['magnitudes']
                magnitudes_right = trace['magnitudes_right']
                units, units_right = trace['units'], trace['units_right']
                ymin, ymax = trace['ymin'], trace['ymax']
                ymin_right, ymax_right = trace['ymin_right'], trace['ymax_right']
                xmin, xmax, logx = trace['xmin'], trace['xmax'], trace['logx']

                if num_samples > 0:
                    freqs = freqs[:num_samples]
                    magnitudes = magnitudes[:num_samples]
                    if isinstance(magnitudes_right, np.ndarray):
                        magnitudes_right = magnitudes_right[:num_samples]
                    elif isinstance(magnitudes_right, dict):
                        magnitudes_right = {k: v[:num_samples] for k, v in magnitudes_right.items()}

                ax.plot(freqs, magnitudes, label=label)
                ax.set_title(label)
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel(f'Magnitude ({units})')
                if ymin is not None and ymax is not None:
                    ax.set_ylim(ymin, ymax)
                if xmin is not None and xmax is not None:
                    ax.set_xlim(xmin, xmax)
                if logx:
                    ax.set_xscale('log')

                if magnitudes_right is not None:
                    ax_right = ax.twinx()
                    if isinstance(magnitudes_right, dict):
                        for sublabel, arr in magnitudes_right.items():
                            ax_right.plot(freqs, arr, linestyle="--", label=sublabel)
                        ax_right.legend(loc="best")
                    else:
                        ax_right.plot(freqs, magnitudes_right, 'r', label=f'{label} Right')

                    ax_right.set_ylabel(f'Magnitude ({units_right})', color='r')
                    if ymin_right is not None and ymax_right is not None:
                        ax_right.set_ylim(ymin_right, ymax_right)

        if self.main_title:
            fig.suptitle(self.main_title, fontsize=self.main_title_fontsize)
            fig.subplots_adjust(top=0.95)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=block)

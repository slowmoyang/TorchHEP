from typing import Union, Optional
from collections.abc import Sequence
from pathlib import Path
from enum import Enum
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_learning_curve(reader,
                        output_dir: Path,
                        metric: str,
                        phase: Union[str, Sequence[str], Enum, Sequence[Enum]],
                        ylabel: Optional[str] = None,
                        max_xticks: int = 5,
) -> None:
    r"""
    """
    fig, ax = plt.subplots()

    train_color, val_color, test_color, *_ = mpl.colors.TABLEAU_COLORS.keys()
    test_marker = '*'

    phase_list = [phase] if not isinstance(phase, Sequence) else phase
    for phase in phase_list:
        if isinstance(phase, Enum):
            phase = phase.value
        tag = f'{metric}/{phase}'

        if phase.lower() == 'training':
            curve = reader.read_scalars(tag)
            ax.plot(curve.step, curve.value, color=train_color,
                    alpha=0.5, lw=2)

            # smooth
            smooth_curve = lowess(endog=curve.value, exog=curve.step,
                                  frac=0.075, it=0, is_sorted=True)
            ax.plot(smooth_curve[:, 0], smooth_curve[:, 1],
                    color=train_color, lw=3)

        elif phase.lower() == 'validation':
            curve = reader.read_scalars(tag)
            ax.plot(curve.step, curve.value, color=val_color, lw=3,
                    ls='--')

        elif phase.lower() == 'test':
            point = reader.read_single_scalar(tag)
            ax.plot(point.step, point.value, label=phase, color=test_color,
                    marker=test_marker, markersize=40, ls='')
            ax.axvline(point.step, color=test_color, ls='--', alpha=0.5)
            ax.axhline(point.value, color=test_color, ls='--', alpha=0.5)
        else:
            raise ValueError(phase)

    # set tick labels
    epoch_data = reader.read_scalars('epoch')
    steps = epoch_data.step.values
    epochs = epoch_data.value.values.astype(int)
    num_points = len(steps)
    if num_points > max_xticks:
        # almost evenly spaced indices including first and last
        spacing = (num_points - 1) // (max_xticks - 1)
        indices = list(range(0, num_points - spacing, spacing)) + [-1]
        steps = steps[indices]
        epochs = epochs[indices]
    _ = ax.set_xticks(steps)
    _ = ax.set_xticklabels(epochs)
    ax.set_xlabel('Epoch')

    # more
    ax.legend()
    ax.set_ylabel(ylabel or metric)
    ax.grid()

    # save figures
    for yscale in ['linear', 'log']:
        ax.set_yscale(yscale)
        fig.tight_layout()
        name = f'{metric}__logy' if yscale == 'log' else metric
        for suffix in ['.png', '.pdf']:
            fig_path = output_dir.joinpath(name).with_suffix(suffix)
            fig.savefig(fig_path)

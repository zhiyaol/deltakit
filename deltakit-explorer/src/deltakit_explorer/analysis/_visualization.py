import matplotlib.pyplot as plt
from typing import Sequence
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
from deltakit_core.constants import RIVERLANE_COLORS_LIST as colors

def plot_leppr(
    num_failed_shots: npt.NDArray[np.int_] | Sequence[int],
    num_shots: npt.NDArray[np.int_] | Sequence[int],
    num_rounds: npt.NDArray[np.int_] | Sequence[int],
    estimated_leppr: float,
    estimated_leppr_stddev: float,
    estimated_spam_error: float,
    estimated_spam_error_stddev: float,
    interpolation_points: int = 26,
    fig: Figure | None = None,
    ax: Axes | None = None,
    plot_error_band: bool = False,
) -> tuple[Figure, Axes]:
    """Plot the logical error probability per round data and the fitted curve.

    Args:
        num_failed_shots (npt.NDArray[np.int_] | Sequence[int]):
            a sequence of integers representing the number of failed shots for each of
            the number of rounds in ``num_rounds``. Should be the same length as
            ``num_rounds``.
        num_shots (npt.NDArray[np.int_] | Sequence[int]):
            a sequence of integers representing the total number of shots for each of
            the number of rounds in ``num_rounds``. Should be the same length as
            ``num_rounds``.
        num_rounds (npt.NDArray[np.int_] | Sequence[int]):
            a sequence of integers representing the number of rounds used to get the
            corresponding results in ``num_failed_shots`` and ``num_shots``. 
        estimated_leppr (float):
            the fitted logical error probability per round
        estimated_leppr_stddev (float):
            the standard deviation of the fitted logical error probability per round
        estimated_spam_error (float):
            the fitted state preparation and measurement error
        estimated_spam_error_stddev (float):
            the standard deviation of the fitted state preparation and measurement error
        interpolation_points (int, optional):
            the number of points to use for plotting the fitted curve. Default is 26.
        fig (Figure | None, optional):
            a matplotlib Figure object to plot on. If None, a new figure will be created.
            Default is None.
        ax (Axes | None, optional):
            a matplotlib Axes object to plot on. If None, a new axes will be created.
            Default is None.
        plot_error_band (bool, optional):
            whether to plot the error band for the fitted curve. Default is False.

    Returns:
        tuple[Figure, Axes]: The matplotlib Figure and Axes objects containing the plot.

    Example
        plot_leppr(
            num_failed_shots=[34, 151, 356],
            num_shots=[500000] * 3,
            num_rounds=[2, 4, 6],
            estimated_leppr=0.00018107956282203963,
            estimated_leppr_stddev=0.017736195386044953,
            estimated_spam_error=0,
            estimated_spam_error_stddev=0.09464885365174087,
        )

    """
    assert not (fig is None) ^ (ax is None)
    if fig is None and ax is None:
        fig, ax = plt.subplots()
    assert fig is not None and ax is not None

    # Get the inputs as numpy arrays.
    # Sanitization: also make sure that the inputs are sorted.
    isort = np.argsort(num_rounds)
    num_rounds = np.asarray(num_rounds)[isort]
    num_failed_shots = np.asarray(num_failed_shots)[isort]
    num_shots = np.asarray(num_shots)[isort]

    logical_error_rates = num_failed_shots / num_shots
    logical_error_rates_stddev = np.sqrt(logical_error_rates * (1 - logical_error_rates) / num_shots)

    # Plotting
    plt.errorbar(num_rounds, logical_error_rates, yerr=logical_error_rates_stddev, fmt="o", color=colors[0])

    rounds_interpolated = np.linspace(
        num_rounds[0], num_rounds[-1], interpolation_points,
        dtype=np.float64,
    )
    y_interpolated = [(1 - 2 * estimated_spam_error) * (1 - 2 * estimated_leppr) ** r for r in rounds_interpolated]
    # if an error probability is below 0, set to 0 for log scale plot performace
    y_interpolated = [0 if i < 0 else i for i in y_interpolated]

    LEP_interpolated = (1.0 - np.array(y_interpolated, dtype=np.float64)) * 0.5
    ax.plot(rounds_interpolated, LEP_interpolated,label=f"Fit, ε={estimated_leppr:.4f}" + r"$\pm$" + f"{estimated_leppr_stddev:.4f}", color=colors[0])

    # add error band
    if plot_error_band:
        LEP_interpolated_err = np.array([np.sqrt(((2*estimated_spam_error_stddev)/(1-2*estimated_spam_error_stddev))**2 + ((2*estimated_leppr_stddev)/(1-2*estimated_leppr_stddev))**2) for r in rounds_interpolated])
        plt.fill_between(rounds_interpolated, LEP_interpolated - LEP_interpolated_err, LEP_interpolated + LEP_interpolated_err, color=colors[0], alpha=0.2)

    ax.set_xlabel("Rounds")
    ax.set_ylabel("Logical error probability")
    ax.legend()
    #ax.set_yscale("log")
    plt.show()

    return fig, ax

def plot_lambda():
    pass

if __name__ == "__main__":
    plot_leppr(
            num_failed_shots=[34, 151, 356],
            num_shots=[500000] * 3,
            num_rounds=[2, 4, 6],
            estimated_leppr=0.00018107956282203963,
            estimated_leppr_stddev=0.017736195386044953,
            estimated_spam_error=-0.00038273670070532173,
            estimated_spam_error_stddev=0.09464885365174087,
            plot_error_band=True,
        )

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from typing import List, Tuple


TOP_LEVEL = Path(__file__).parent
CONFIGS = TOP_LEVEL / "configs"


def acc(correct: int, total: int) -> float:
    return 100 * float(correct) / float(total)


def simple_scatter_plot(
    data: List[Tuple[float, float]],
    title: str,
    x_label: str = "Iteration",
    y_label: str = "Accuracy",
) -> Figure:
    fig, ax = plt.subplots()
    ax.scatter(*zip(*data))
    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    return fig

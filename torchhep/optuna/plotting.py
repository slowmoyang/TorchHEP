import os
from typing import Optional
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import optuna


def optuna_cosmetics(func):

    def wrapped_func(study, directory=None, **kwargs):
        try:
            ax = func(study, directory=directory, **kwargs)
        except Exception as exception:
            warnings.warn(str(exception), RuntimeWarning)
            return

        if isinstance(ax, np.ndarray):
            axes = ax
            fig = axes.ravel()[0].figure
            fig.set_figwidth(20)
            fig.set_figheight(20)

            fig._suptitle.set_size(30)
            fig._suptitle.set_y(0.9)

            for ax in axes.ravel():
                ax.xaxis.label.set_size(20)
                ax.yaxis.label.set_size(20)
                ax.tick_params(axis='both', labelsize=15)

                if ax.legend_ is not None:
                    ax.legend(prop={'size': 20})
        else:
            fig = ax.figure
            fig.set_figwidth(10)
            fig.set_figheight(10)

            ax.title.set_fontsize(20)

            ax.xaxis.label.set_size(20)
            ax.yaxis.label.set_size(20)
            ax.tick_params(axis='both', labelsize=20)

            if ax.legend_ is not None:
                ax.legend(prop={'size': 20})

        if fig.axes[-1].get_label() == '<colorbar>':
            colorbar = fig.axes[-1]
            colorbar.yaxis.label.set_size(20)
            colorbar.tick_params(axis='y', labelsize=15)

        name = func.__name__
        if name.startswith('plot_'):
            name = name[5:]
        else:
            warnings.warn(
                message=(f"expect a function name, which starts with 'plot_'",
                         f"but got '{name}'"),
                category=UserWarning)

        if directory is not None:
            if 'params' in kwargs:
                name += '-'.join(kwargs['params'])

            for ext in ['png', 'pdf']:
                path = os.path.join(directory, f'{name}.{ext}')
                fig.savefig(path, format=ext, bbox_inches="tight")
        return ax
    return wrapped_func


@optuna_cosmetics
def plot_contour(study, params=None, directory=None, objective_name=None):
    # NOTE scipy.spatial.qhull.QhullError: QH6214 qhull input error: not enough points(2) to construct initial simplex (need 4)
    # TODO try except
    ax = optuna.visualization.matplotlib.plot_contour(study, params)

    if objective_name is not None:
        colorbar = plt.gca().figure.axes[-1]
        if colorbar.get_label() == '<colorbar>':
            ylabel = colorbar.yaxis.label.get_text() + f' ({objective_name})'
            colorbar.yaxis.label.set_text(ylabel)
        else:
            warnings.warn(f'expected a colorbar but got "{colorbar.get_label()}"', RuntimeWarning)

    return ax

@optuna_cosmetics
def plot_edf(study, directory=None, objective_name=None):
    ax = optuna.visualization.matplotlib.plot_edf(study)
    if objective_name is not None:
        xlabel = ax.get_xlabel() + f' ({objective_name})'
        ax.set_xlabel(xlabel)
    return ax

@optuna_cosmetics
def plot_intermediate_values(study, directory=None):
    ax = optuna.visualization.matplotlib.plot_intermediate_values(study)
    return ax

@optuna_cosmetics
def plot_optimization_history(study, directory=None, objective_name=None):
    ax = optuna.visualization.matplotlib.plot_optimization_history(study)
    if objective_name is not None:
        ylabel = ax.get_ylabel() + f' ({objective_name})'
        ax.set_ylabel(ylabel)
    return ax

@optuna_cosmetics
def plot_parallel_coordinate(study, params=None, directory=None):
    ax = optuna.visualization.matplotlib.plot_parallel_coordinate(study, params)
    return ax

@optuna_cosmetics
def plot_param_importances(study, directory=None, objective_name=None):
    ax = optuna.visualization.matplotlib.plot_param_importances(study)
    if objective_name is not None:
        xlabel = ax.get_xlabel() + f' ({objective_name})'
        ax.set_xlabel(xlabel)
    return ax

@optuna_cosmetics
def plot_pareto_front(study, directory=None):
    ax = optuna.visualization.matplotlib.plot_pareto_front(study)
    return ax

@optuna_cosmetics
def plot_slice(study, params=None, directory=None):
    ax = optuna.visualization.matplotlib.plot_slice(study, params)
    return ax

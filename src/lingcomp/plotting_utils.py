import logging

import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)


def create_lineplot(df, path, y, x, y_label=None, x_label=None, hue=None):
    sns_plot = sns.relplot(x=x, y=y, hue=hue, kind="line", data=df)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.xlabel(y_label)
    save_plot(path, type="sns", plot_obj=sns_plot)
    plt.clf()


def create_heatmap(df, path, mask, y_label=None, x_label=None):
    with sns.axes_style("white"):
        heatmap = sns.heatmap(df, mask=mask, vmin=0, vmax=1, annot=True)
        if x_label is not None:
            plt.xlabel(x_label)
        if y_label is not None:
            plt.xlabel(y_label)
        save_plot(path, type="sns", plot_obj=heatmap.get_figure())
        plt.clf()


def save_plot(path, type, plot_obj=None):
    plot_obj.savefig(path)
    logger.info(f"Similarity plot saved in {path}")

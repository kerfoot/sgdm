import matplotlib.pyplot as plt
import seaborn as sns


def plot_distribution(data, num_bins=20, style='white', palette='muted', color_codes=True, figsize=(8,8)):
    """Standardized Histogram/distribution plots of the values contained in data.  4 axes are created.

    Parameters:
        data: (n,1) or (1,n) numpy list of values
        num_bins: number of bins for grouping
        style: seaborn figure/axes style
        palette: color palett name
        color_codes: map matplotlib color abbreviations to the current palette,
        figsize: tuple specifying the figure size

    Returns:
        fig, ax: Figure and list of axes handles"""

    sns.set(style=style, palette=palette, color_codes=color_codes)

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    # Plot a simple histogram with bin size determined automatically
    sns.distplot(data, bins=num_bins, kde=False, color="b", ax=axes[0, 0])

    # Plot a kernel density estimate and rug plot
    sns.distplot(data, bins=num_bins, hist=False, rug=True, color="r", ax=axes[0, 1])

    # Plot a filled kernel density estimate
    sns.distplot(data, bins=num_bins, hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])

    # Plot a histogram and kernel density estimate
    sns.distplot(data, bins=num_bins, color="m", ax=axes[1, 1])

    plt.setp(axes, yticks=[])

    sns.set()

    return fig, axes

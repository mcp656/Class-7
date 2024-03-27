import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import seaborn as sns

def calculate_returns(data: pd.DataFrame):
    """
    Calculates the monthly and cumulative returns for a DataFrame of stock prices

    Parameters:
    data       (pd.DataFrame): DataFrame of stock prices

    Returns:
    data_r     (pd.DataFrame): DataFrame of stock returns
    data_cr    (pd.DataFrame): DataFrame of cumulative stock returns
    """
    # Calculate monthly returns using the pct_change() function
    data_r = data.pct_change()

    # Calculate cumulative returns using the cumprod() function
    data_cr = (1 + data_r).cumprod()

    # Returns monthly and cumulative stock returns
    return data_r, data_cr


def calculate_portfolio_returns(data: pd.DataFrame, weights: pd.DataFrame):
    """
    Calculates portfolio and cumulative returns based on returns and weights for each stock

    Parameters:
    data        (pd.DataFrame): DataFrame of stock returns
    weights     (pd.DataFrame): DataFrame of weights for each stock

    Returns:
    port_r         (pd.Series): Series of portfolio returns
    cum_port_r     (pd.Series): Series of cumulative portfolio returns
    """
    # Calculate weighted returns
    weighted_r = data * weights

    # Calculate portfolio return by summing up the weighted returns across all stocks
    port_r = weighted_r.sum(axis = 1)

    # Calculate cumulative return of the portfolio
    cum_port_r = (1 + port_r).cumprod()

    # Return portfolio and cumulative portfolio returns
    return port_r, cum_port_r


def cum_ret_plot(data: pd.DataFrame, stock: str, ref: str, fig: int = 1, ax_data: pd.DataFrame = None):
    """
    Plots the cumulative return of a stock against a reference 'index'

    Parameters:
    data        (pd.DataFrame): DataFrame of cumulative stock returns
    stock                (str): Which stock to plot
    ref                  (str): Reference/Index
    fig                  (int): Figure no. in title, (defualt = 1)
    ax_data     (pd.DataFrame): Only used for non-interactive plots, (default = None).

    Returns:
    A plot of a stocks cumulative return compared to a reference/index
    """
    # If ax_data is None, set it to the input data for an interactive plot
    if ax_data is None:
        ax_data = data

    # Plot the specified stock using ax_data
    ax = ax_data.plot(y = stock)

    # Plot the reference 'index' on the same axes (by specifying ax=ax)
    data.plot(y = ref, ax = ax, 
              title = f'Figure {fig}: Cumulative Return of {stock} compared to {ref}',
              ylabel = 'Cumulative Return')



def plot_scatter_with_labels(ax: plt.Axes, x: list, y: list, labels: list, title: str, xlabel: str, ylabel: str):
    """
    Plots a scatter-plot with automatically adjusted labels for each point, a trend line and labels for the title and axes

    Parameters:
    ax          (plt.Axes): The Axes object to draw the plot onto
    x               (list): List of x-axis values
    y               (list): List of y-axis values
    labels          (list): List of labels for the data points
    title            (str): The title for the graph
    xlabel           (str): The label for the x-axis
    ylabel           (str): The label for the y-axis

    Returns:
    A scatter-plot with automatically adjusted labels for each point, a trend line and labels for the title and axes
    """
    # Create the scatter plot with a trend line
    ax.scatter(x, y)
    sns.regplot(x = x, y = y, scatter = False, ax = ax)

    # Add labels to each data point using ax.text in a for loop
    texts = [ax.text(x_pos, y_pos, lab, fontsize = 8, ha = 'center') for x_pos, y_pos, lab in zip(x, y, labels)]

    # Automatically adjust the labels to avoid overlapping using adjust_text
    adjust_text(texts, arrowprops=dict(arrowstyle = '-', color = 'k', lw = 0.5), ax = ax)

    # Set the title and labels for the axes
    ax.set_title(title, fontsize = 12)
    ax.set_xlabel(xlabel, fontsize = 12)
    ax.set_ylabel(ylabel, fontsize = 12)


def normalize_column(col: pd.Series):
    """
    Normalizes a column to one

    Parameters:
    col        (pd.Series): The column to be normalized

    Returns:
    norm_col   (pd.Series): A normalized column
    """
    # Calculate the sum of the column
    col_sum = col.sum()

    # Normalize each value in the column by dividing it by the sum
    norm_col = col / col_sum

    # Return the normalized column
    return norm_col
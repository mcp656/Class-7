# Imports and installations first
import numpy as np
import matplotlib.pyplot as plt

# Then functions
def x_squared(x):
    """
    Compute the square of each element in the input array.

    Parameters:
    x (array-like): An array of numeric values for which the square will be computed.

    Returns:
    array-like: An array containing the square of each element from the input array.
    """
    # Return x squared
    return x**2

def simple_fig(x, y, color='blue', label='y'):
    """
    Create a line plot for the given x and y data arrays with customizable color and label.

    Parameters:
    x (array-like): An array of x-values for the plot.
    y (array-like): An array of y-values that correspond to x-values.
    color (str, optional): The color of the line in the plot. Default is 'blue'.
    label (str, optional): The label for the plot data, used in the legend. Default is 'y'.

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot.
    """
    # Create figure
    fig = plt.figure()

    # Add subplot with 1 row, 1 column, and 1 index
    ax = fig.add_subplot(1, 1, 1)

    # Plot the data
    ax.plot(x, y, '--', color=color, label=label)

    # Display legend
    ax.legend()

    # Show plot
    plt.show();
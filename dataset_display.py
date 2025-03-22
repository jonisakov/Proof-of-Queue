import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def calc_avg(all_runs, minimum_value, step=1000):
    """
    Calculates the average number of iterations for FIFO and binary tree
    algorithms for values below the minimum_value threshold.

    Args:
        all_runs (list of tuples): List of (FIFO, Binary Tree) iteration counts.
        minimum_value (int): Minimum value threshold for filtering.

    Returns:
        tuple: Averages for FIFO and binary tree algorithms.
    """
    rand_sum = 0
    turn_sum = 0
    counter = 0

    for nums in all_runs:
        if minimum_value - step <= nums[0] < minimum_value:
            counter += 1
            rand_sum += nums[0]
            turn_sum += nums[1]

    if counter == 0:
        return None, None  # Avoid division by zero

    fifo_avg = rand_sum / counter
    binary_tree_avg = turn_sum / counter

    return fifo_avg, binary_tree_avg


def plot_results_with_regression(all_runs, max_threshold, step=1000):
    """
    Plots the results of calc_avg over a range of threshold values and fits
    a linear regression to the binary tree and FIFO averages.

    Args:
        all_runs (list of tuples): List of (FIFO, Binary Tree) iteration counts.
        max_threshold (int): Maximum threshold value to consider.
        step (int): Step size for thresholds.
    """
    thresholds = []
    fifo_averages = []
    binary_tree_averages = []

    for minimum_value in range(0, max_threshold + 1, step):
        fifo_avg, binary_tree_avg = calc_avg(all_runs, minimum_value)

        if fifo_avg is not None and binary_tree_avg is not None:
            thresholds.append(minimum_value)
            fifo_averages.append(fifo_avg)
            binary_tree_averages.append(binary_tree_avg)

    # Convert data to numpy arrays for regression
    thresholds_np = np.array(thresholds).reshape(-1, 1)
    fifo_averages_np = np.array(fifo_averages)
    binary_tree_averages_np = np.array(binary_tree_averages)

    # Perform linear regression for binary tree
    binary_tree_model = LinearRegression()
    binary_tree_model.fit(thresholds_np, binary_tree_averages_np)
    binary_tree_predicted = binary_tree_model.predict(thresholds_np)

    # Perform linear regression for FIFO
    fifo_model = LinearRegression()
    fifo_model.fit(thresholds_np, fifo_averages_np)
    fifo_predicted = fifo_model.predict(thresholds_np)

    # Print gradients and intercepts
    print(f"Binary Tree Regression: Gradient = {binary_tree_model.coef_[0]}, Intercept = {binary_tree_model.intercept_}")
    print(f"FIFO Regression: Gradient = {fifo_model.coef_[0]}, Intercept = {fifo_model.intercept_}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, fifo_averages, 'o-', label='FIFO Average', color='blue')
    plt.plot(thresholds, binary_tree_averages, 'o-', label='Binary Tree Average', color='green')
    plt.plot(thresholds, binary_tree_predicted, '--', label='Binary Tree Regression', color='red')
    plt.plot(thresholds, fifo_predicted, '--', label='FIFO Regression', color='orange')

    plt.xlabel('Minimum Value Threshold')
    plt.ylabel('Average Number of Iterations')
    plt.title('Average Iterations vs. Threshold with Linear Regression')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage
all_runs = resualts

plot_results_with_regression(all_runs, max_threshold=66000, step=1000)
len(all_runs)
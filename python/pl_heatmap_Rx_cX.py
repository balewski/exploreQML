import numpy as np
import matplotlib.pyplot as plt
import os

def plot_heatmap(outPath, function, file_name):
    # Create a grid of values for p0 and p1
    p0 = np.linspace(-1, 1, 100)
    p1 = np.linspace(-1, 1, 100)
    P0, P1 = np.meshgrid(p0, p1)

    # Calculate the function values and get the LaTeX formula
    F, latex_formula = function(P0, P1)

    # Create the plot with square aspect ratio
    fig, ax = plt.subplots(figsize=(6, 6))  # Square aspect ratio
    c = ax.pcolormesh(P0, P1, F, cmap='viridis', shading='auto')
    ax.set_title('f(.)= ${}$'.format(latex_formula))
    ax.set_xlabel('$p_0$')
    ax.set_ylabel('$p_1$')
    ax.set_aspect('equal', adjustable='box')  # Set aspect ratio to 1
    fig.colorbar(c, ax=ax)

    # Ensure the output directory exists
    if not os.path.exists(outPath):
        os.makedirs(outPath)

    # Save the plot as a PNG file
    plt.savefig(os.path.join(outPath, file_name))

def f2(p0, p1):
    function_values = 0.5 * (p0 * np.abs(1 + p1) + (1 - p0) * np.abs(1 - p1))
    latex_formula = r'0.5(p_0|1+p_1| + (1-p_0)|1-p_1|)'
    return function_values, latex_formula


def f_I(x0):
    return np.abs(1 - x0)/2

def f_H(x0):
    # Calculate the complex part using sqrt(1 - x0)
    complex_part = np.sqrt(1 - x0) * 1j  # j is the imaginary unit in Python
    return np.abs((np.sqrt(1 + x0) + complex_part) / 2)

def f_3(x0, x1):
    # Calculate p1 using the formula provided
    p1 = (1 - x1) / 2
    return p1 * f_I(x0) + (1 - p1) * f_H(x0)

def f3(x0, x1):
    function_values = f_3(x0, x1)
    latex_formula = r'p_1 \cdot f_I(x_0) + (1 - p_1) \cdot f_H(x_0)'
    return function_values, latex_formula

if __name__ == "__main__":
    outPath = 'output_plots'  # Define the output directory
    #1plot_heatmap(outPath, f2, 'f2_heatmap.png')
    plot_heatmap(outPath, f3, 'f3_heatmap.png')

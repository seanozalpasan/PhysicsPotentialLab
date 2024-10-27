import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

def visualize_electric_field(csv_path):
    """
    Creates a visualization of electric field lines and equipotential lines
    from a CSV file containing electric potential values.
    
    Args:
        csv_path (str): Path to the CSV file containing the electric potential data
    """
    # Read the CSV file
    potential_data = pd.read_csv(csv_path, header=None).values
    
    # Create coordinate grids for the original data
    y_orig = np.arange(potential_data.shape[0])
    x_orig = np.arange(potential_data.shape[1])
    
    # Calculate electric field components using gradient
    grad_y, grad_x = np.gradient(potential_data)
    Ey, Ex = -grad_y, -grad_x
    
    # Calculate field magnitude (proportional to density of field lines)
    E_magnitude = np.sqrt(Ex**2 + Ey**2)
    
    # Create interpolators for Ex, Ey, and density
    Ex_interp = RegularGridInterpolator((y_orig, x_orig), Ex, bounds_error=False, fill_value=None)
    Ey_interp = RegularGridInterpolator((y_orig, x_orig), Ey, bounds_error=False, fill_value=None)
    density_interp = RegularGridInterpolator((y_orig, x_orig), E_magnitude, bounds_error=False, fill_value=None)
    
    # Create the figure and axis
    plt.figure(figsize=(12, 10))
    
    # Plot equipotential lines with increased thickness
    contours = plt.contour(x_orig, y_orig, potential_data, levels=15, colors='blue', alpha=0.6, linewidths=1.5)
    plt.clabel(contours, inline=True, fontsize=8, fmt='%.1f V')
    
    # Create grid for streamplot
    nx, ny = 30, 30  # Number of points in each direction
    x_grid = np.linspace(0, potential_data.shape[1]-1, nx)
    y_grid = np.linspace(0, potential_data.shape[0]-1, ny)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Create points for interpolation
    points = np.array([(y, x) for y in y_grid for x in x_grid]).reshape(ny, nx, 2)
    
    # Interpolate vector field onto regular grid
    Ex_grid = Ex_interp(points).reshape(ny, nx)
    Ey_grid = Ey_interp(points).reshape(ny, nx)
    density_grid = density_interp(points).reshape(ny, nx)
    
    # Normalize the density grid
    density_grid = (density_grid - density_grid.min()) / (density_grid.max() - density_grid.min())
    
    # Plot electric field lines using streamplot with increased thickness
    # Increased base linewidth from 1 to 2.5 and adjusted density
    plt.streamplot(x_grid, y_grid, Ex_grid, Ey_grid,
                  density=1.5,
                  linewidth=2.5*density_grid,  # Increased base linewidth
                  color='red',
                  arrowsize=1.5,  # Slightly increased arrow size
                  integration_direction='both')
    
    # Add colormap of potential with reduced alpha for better line visibility
    plt.imshow(potential_data, extent=[0, potential_data.shape[1]-1, 
                                     potential_data.shape[0]-1, 0],
              cmap='viridis', alpha=0.2)  # Reduced background alpha for better contrast
    plt.colorbar(label='Electric Potential (V)')
    
    # Customize the plot
    plt.title('Electric Field and Equipotential Lines', fontsize=14)
    plt.xlabel('X Position', fontsize=12)
    plt.ylabel('Y Position', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    return plt

# Example usage with your CSV file
csv_file_path = "labdata.csv"
plt = visualize_electric_field(csv_file_path)
plt.show()
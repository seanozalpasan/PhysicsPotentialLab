import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd

def visualize_electric_field(csv_path):
    """
    Creates a visualization of electric field lines and equipotential lines
    from a CSV file containing electric potential values.
    
    Args:
        csv_path (str): Path to the CSV file containing the electric potential data
    """
    # Read the CSV file
    # Using pd.read_csv with header=None since your data doesn't have headers
    potential_data = pd.read_csv(csv_path, header=None).values
    
    # Create coordinate grids
    y, x = np.mgrid[0:potential_data.shape[0], 0:potential_data.shape[1]]
    
    # Calculate electric field components using gradient
    # Negative gradient because E = -∇V
    grad_y, grad_x = np.gradient(potential_data)
    Ey, Ex = -grad_y, -grad_x  # Apply negative sign separately
    
    # Normalize the electric field vectors for better visualization
    E_magnitude = np.sqrt(Ex**2 + Ey**2)
    Ex = Ex / (E_magnitude + 1e-10)  # Add small number to avoid division by zero
    Ey = Ey / (E_magnitude + 1e-10)
    
    # Create the figure and axis
    plt.figure(figsize=(12, 10))
    
    # Plot equipotential lines
    contours = plt.contour(x, y, potential_data, levels=15, colors='blue', alpha=0.6)
    plt.clabel(contours, inline=True, fontsize=8, fmt='%.1f V')
    
    # Plot electric field lines using streamplot
    # Flip y-axis for correct orientation (0,0 at top left)
    plt.streamplot(x, y, Ex, Ey, color='red', density=1.5, linewidth=1, 
                  arrowsize=1.5, integration_direction='both')
    
    # Add colormap of potential
    plt.imshow(potential_data, extent=[0, potential_data.shape[1]-1, 
                                     potential_data.shape[0]-1, 0],
              cmap='viridis', alpha=0.3)
    plt.colorbar(label='Electric Potential (V)')
    
    # Customize the plot
    plt.title('Electric Field and Equipotential Lines')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True, alpha=0.3)
    
    return plt

# Example usage with your CSV file
csv_file_path = "labdata.csv"  # Update this to your CSV file path
plt = visualize_electric_field(csv_file_path)
plt.show()
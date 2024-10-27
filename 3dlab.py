import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def load_voltage_data(file_path):
    """Load voltage data from CSV file and convert to numpy array"""
    try:
        voltage_data = pd.read_csv(file_path, header=None).values
        return voltage_data
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def calculate_electric_field(voltage_data, dx=1, dy=1):
    """Calculate electric field components using gradient"""
    Ey, Ex = np.gradient(-voltage_data, dy, dx)
    return Ex, Ey

def plot_3d_field(voltage_data, num_field_lines=20):
    """Create 3D visualization of electric field"""
    # Create coordinate meshgrid
    rows, cols = voltage_data.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Calculate electric field
    Ex, Ey = calculate_electric_field(voltage_data)
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(x, y, voltage_data, 
                          cmap='RdYlBu_r',
                          alpha=0.8,
                          linewidth=0,
                          antialiased=True)
    
    # Plot contour lines on the surface
    contours = ax.contour(x, y, voltage_data,
                         zdir='z',
                         offset=np.min(voltage_data),
                         levels=15,
                         cmap='Blues',
                         alpha=0.6)
    
    # Add field line arrows at regular intervals
    step = max(rows//10, cols//10)  # Adjust arrow density
    arrow_length = 0.5
    
    for i in range(0, rows, step):
        for j in range(0, cols, step):
            # Normalize electric field components at this point
            Ex_norm = Ex[i, j] / (np.sqrt(Ex[i, j]**2 + Ey[i, j]**2) + 1e-10)
            Ey_norm = Ey[i, j] / (np.sqrt(Ex[i, j]**2 + Ey[i, j]**2) + 1e-10)
            
            # Plot arrow
            ax.quiver(j, i, voltage_data[i, j],  # Start point
                     arrow_length * Ex_norm,  # x-direction
                     arrow_length * Ey_norm,  # y-direction
                     0,  # z-direction
                     color='red',
                     alpha=0.6,
                     length=0.5,
                     normalize=True)
    
    # Customize the plot
    ax.set_title('3D Electric Potential Surface with Field Lines', pad=20)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Electric Potential (V)')
    
    # Move axes to match 2D orientation
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    
    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Electric Potential (V)')
    
    # Adjust viewing angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    return plt

def visualize_electric_field_3d(file_path):
    """Main function to create 3D visualization from CSV file"""
    # Load data
    voltage_data = load_voltage_data(file_path)
    
    if voltage_data is None:
        return
    
    # Apply slight smoothing to reduce noise
    voltage_data = gaussian_filter(voltage_data, sigma=0.5)
    
    # Create visualization
    plt = plot_3d_field(voltage_data)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Example usage
    file_path = "labdata.csv"  # Replace with your CSV file path
    visualize_electric_field_3d(file_path)
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd

def load_voltage_data(file_path):
    """Load voltage data from CSV file and convert to numpy array"""
    try:
        # Read CSV file without headers
        voltage_data = pd.read_csv(file_path, header=None).values
        return voltage_data
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def calculate_electric_field(voltage_data, dx=1, dy=1):
    """Calculate electric field components using gradient"""
    Ey, Ex = np.gradient(-voltage_data, dy, dx)  # Negative gradient gives E-field
    return Ex, Ey

def normalize_vectors(Ex, Ey):
    """Normalize electric field vectors"""
    E_magnitude = np.sqrt(Ex**2 + Ey**2)
    Ex_norm = Ex / (E_magnitude + 1e-10)  # Add small number to avoid division by zero
    Ey_norm = Ey / (E_magnitude + 1e-10)
    return Ex_norm, Ey_norm

def plot_field_and_equipotential(voltage_data, num_field_lines=20, num_equipotential=15):
    """Create visualization of electric field and equipotential lines"""
    # Create coordinate meshgrid with y increasing downward
    rows, cols = voltage_data.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Calculate electric field
    Ex, Ey = calculate_electric_field(voltage_data)
    Ex_norm, Ey_norm = normalize_vectors(Ex, Ey)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create filled contour plot for the background
    contourf = plt.contourf(x, y, voltage_data, levels=50, 
                           cmap='RdYlBu_r', alpha=0.3)
    
    # Plot equipotential lines
    contour = plt.contour(x, y, voltage_data, levels=num_equipotential, 
                         colors='blue', alpha=0.8)
    plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f V')
    
    # Calculate field strength for density scaling
    E_magnitude = np.sqrt(Ex**2 + Ey**2)
    max_magnitude = np.max(E_magnitude)
    
    # Use a scalar density value that varies with field strength
    density = 1.5  # Base density value
    
    # Plot electric field lines using streamplot
    plt.streamplot(x, y, Ex_norm, Ey_norm, 
                  density=density,
                  color='red',
                  linewidth=1,
                  arrowsize=1,
                  arrowstyle='->',
                  minlength=0.3)
    
    # Move x-axis to top
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    
    # Customize plot
    plt.title('Electric Field Lines and Equipotential Lines', pad=25)  # Add padding to title
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.colorbar(contourf, label='Electric Potential (V)')
    plt.grid(True, alpha=0.3)
    
    # Invert y-axis to make (0,0) at top-left
    plt.gca().invert_yaxis()
    
    return plt

def visualize_electric_field(file_path):
    """Main function to create visualization from CSV file"""
    # Load data
    voltage_data = load_voltage_data(file_path)
    
    if voltage_data is None:
        return
    
    # Apply slight smoothing to reduce noise
    voltage_data = gaussian_filter(voltage_data, sigma=0.5)
    
    # Create visualization
    plt = plot_field_and_equipotential(voltage_data)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Example usage
    file_path = "labdata.csv"  # Replace with your CSV file path
    visualize_electric_field(file_path)
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# Set A4 size in inches
a4_width, a4_height = 8.27, 11.69  

# Sample Data
x = np.linspace(0, 10, 100)
y1, y2, y3, y4, y5, y6, y7 = np.sin(x), np.cos(x), np.tan(x), np.exp(-x), np.log1p(x), np.sqrt(x), x**2

# Create figure with A4 dimensions
fig = plt.figure(figsize=(a4_width, a4_height), dpi=300)

# Define GridSpec layout with custom row heights
gs = gridspec.GridSpec(3, 3, height_ratios=[1.5, 1, 1])  # Custom row heights

# Row 1: 3 Plots (Each takes one column)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

# Row 2: 2 Plots (First plot spans 2 columns)
ax4 = fig.add_subplot(gs[1, :2])  # Spanning first 2 columns
ax5 = fig.add_subplot(gs[1, 2])   # Single column

# Row 3: 2 Plots (First plot spans 2 columns)
ax6 = fig.add_subplot(gs[2, :2])  # Spanning first 2 columns
ax7 = fig.add_subplot(gs[2, 2])   # Single column

# Plot Data
ax1.plot(x, y1, 'r'); ax1.set_title("Sine Function")
ax2.plot(x, y2, 'b'); ax2.set_title("Cosine Function")
ax3.plot(x, y3, 'g'); ax3.set_title("Tangent Function")
ax4.plot(x, y4, 'm'); ax4.set_title("Exponential Decay")
ax5.plot(x, y5, 'c'); ax5.set_title("Log Function")
ax6.plot(x, y6, 'y'); ax6.set_title("Square Root")
ax7.plot(x, y7, 'k'); ax7.set_title("Quadratic Function")
fig.suptitle("Station : AGRI")
# Adjust layout to fit A4
plt.tight_layout()

# Save as high-quality A4-sized PDF & PNG
fig.savefig("A4_subplots_figure.pdf", format='pdf', dpi=300, bbox_inches='tight')


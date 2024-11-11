

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Create a radial gradient
def radial_gradient(radius):
    size = 10 * radius
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)*4
    mask = np.clip(1 - distance, 0, 1)  # Fade out to edges

    return mask * 200 # Scale color by the mask intensity

  # Maximum color intensity for gradient, use a scalar value

center_x, center_y = 5, 5

# Generate the gradient mask
gradient_circle = radial_gradient(10)

# Plotting
fig, ax = plt.subplots()

cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "#9b181eff"])

# Display the gradient as an image, adjust position and scaling with extent
extent = (center_x - 1, center_x + 1, center_y - 1, center_y + 1)
ax.imshow(gradient_circle, extent=extent, origin="lower", cmap=cmap)

# Additional plot configurations
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
plt.show()
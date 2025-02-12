import numpy as np
import matplotlib.pyplot as plt

# Define the data for each feature (example data)
data = [0.2, 0.4, 0.6, 0.8, 0.5, 0.7, 0.3]
num_vars = len(data)

# Repeat the first value to close the circle
values = data + data[:1]

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# Create the radar chart
fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))

# Draw one axe per variable and add labels
ax.set_theta_offset(np.pi / num_vars)
ax.set_theta_direction(-1)

# Draw the outline of our data and Fill in the area of our data
ax.plot(angles, values, linewidth=1, linestyle='solid')
ax.fill(angles, values, 'b', alpha=0.1)

# Add labels for each feature
ax.set_xticks(angles[:-1])
ax.set_xticklabels(['kNN acc', 'SVM acc', 'JS div', 'Acc_r', 'Acc_f', 'cm_diff', 'l2 weight'])

# hide the number labels
ax.set_yticklabels([])
macro_labels = ['Architecture Similarity', 'Feature Space Similarity', 'Prediction Similarity']

def paint_sector(ax, start_angle, end_angle, color):
    theta = np.linspace(start_angle, end_angle, 100)
    r = np.ones_like(theta)
    ax.fill_between(theta, r, color=color, alpha=0.15)

# Example usage of paint_sector with half-sector offset
half_sector = (angles[1] - angles[0]) / 2
paint_sector(ax, angles[0] - half_sector, angles[3] - half_sector, 'yellow') 
paint_sector(ax, angles[3] - half_sector, angles[6] - half_sector, 'green')
paint_sector(ax, angles[6] - half_sector, angles[6] + half_sector, 'orange')




plt.show()
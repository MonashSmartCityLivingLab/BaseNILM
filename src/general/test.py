import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Create a figure and an axis
fig, ax = plt.subplots()

# Plot data
ax.plot(x, y, marker='o')

# Set title and labels
ax.set_title('Basic Line Plot')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

# Show the plot
print("Running")
plt.show()
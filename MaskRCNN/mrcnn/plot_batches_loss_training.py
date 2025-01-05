import matplotlib.pyplot as plt

# Data for plotting
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Y = [3.65422024, 4.4349, 4.1571, 3.9412, 3.8626, 3.7743, 3.7358, 3.6632, 3.6281, 3.5895]

# Plotting the data
plt.plot(X, Y, label='Loss')

# Adding title and labels
plt.title('Loss Over Time')
plt.xlabel('Batches in the first epoch')
plt.ylabel('Loss')

# Adding a legend
plt.legend()

# Display the plot
plt.show()

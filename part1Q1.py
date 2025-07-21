import matplotlib.pyplot as plt
import numpy as np

# Define ReLU and PReLU activation functions
def relu(x):
    return np.maximum(0, x)

def prelu(x, alpha):
    return np.where(x > 0, x, alpha * x)

# Generate x values
x = np.linspace(-5, 5, 100)

# Apply ReLU and PReLU activations
y_relu = relu(x)
y_prelu = prelu(x, alpha=0.5)  # You can choose an arbitrary alpha value

# Plot the functions
plt.figure(figsize=(8, 6))
plt.plot(x, y_relu, label='ReLU', color='blue')
plt.plot(x, y_prelu, label='PReLU (alpha=0.5)', color='red')
plt.xlabel('x')
plt.ylabel('Activation')
plt.legend()
plt.title('ReLU vs Parametric ReLU Activation Functions')
plt.grid(True)
plt.savefig("ReLU vs Parametric ReLU.png")
plt.show()

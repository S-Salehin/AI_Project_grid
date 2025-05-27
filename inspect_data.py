import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/taskA/train.npz")
X, y = data["X"], data["y"]
for i in [0, 10, 42]:
    plt.imshow(X[i], cmap="gray")
    plt.title(f"Label: {y[i]:.2f}")
    plt.show()

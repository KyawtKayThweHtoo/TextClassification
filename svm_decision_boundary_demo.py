import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification

# Generate synthetic 2D data for 5 classes
X, y = make_classification(
    n_samples=100, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, n_classes=5, random_state=42
)

# Fit linear SVM (one-vs-rest)
clf = svm.SVC(kernel='linear', decision_function_shape='ovr')
clf.fit(X, y)

# Create color map
colors = ['red', 'blue', 'green', 'orange', 'purple']
labels = [
    "Artificial Intelligence",
    "Distribution Data",
    "Image Preprocessing",
    "Networking & Cybersecurity",
    "Software Engineering"
]

# Plot the data points
plt.figure(figsize=(8, 6))
for i, color in enumerate(colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=labels[i], edgecolors='k')

# Plot decision boundaries
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 200),
                     np.linspace(ylim[0], ylim[1], 200))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.2, colors=colors, levels=np.arange(-0.5,5,1))

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Linear SVM Decision Boundaries (Example with 5 Categories)')
plt.legend()
plt.show()

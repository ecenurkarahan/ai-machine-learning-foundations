from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np


# with this function, i generated a non linear two class, 2 features dataset.
def generate_dataset():
    # n_samples: number of samples, noise: standard deviation of Gaussian noise added to the data
    # random_state: seed for reproducibility
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    # plotting the dataset that i generated
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')
    plt.title("Non-linear Two-Class Data (make_moons)")
    plt.show()
    return X, y


def plot_decision_boundary(model, X, y, title):
    # Create a grid to evaluate model
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap='bwr', alpha=0.2)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')
    plt.title(title)
    plt.show()


def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Linear SVM": SVC(kernel='linear', C=1.0),
        "Polynomial SVM (degree=3)": SVC(kernel='poly', degree=3, C=1.0),
        "RBF SVM": SVC(kernel='rbf', C=1.0)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        results[name] = (1 - train_acc, 1 - test_acc)  # store error rates

        print(f"{name}:")
        print(f"  → Training error: {1 - train_acc:.3f}")
        print(f"  → Test error:     {1 - test_acc:.3f}\n")

        # plot decision boundary
        plot_decision_boundary(model, X_train, y_train, f"{name} - Decision Boundary")

    return results


def main():
    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("Summary of error rates:")
    for model, (train_err, test_err) in results.items():
        print(f"{model:30s} | Train error: {train_err:.3f} | Test error: {test_err:.3f}")


if __name__ == "__main__":
    main()

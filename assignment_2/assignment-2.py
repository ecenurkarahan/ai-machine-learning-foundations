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
    # here i started by finding the min and max values for both features
    # so that i can create a grid that is centered by the data points
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    # after finding min and max, i created a grid that covers the entire feature space
    # the background canvas has 300*300 points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    # predicting the class for each point in the grid based on the given model
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # we flattened the array to make predictions, we need to reshape it back to the grid shape
    Z = Z.reshape(xx.shape)
    # this contour highlights the decision boundary, colors the regions for each class
    plt.contourf(xx, yy, Z, cmap='bwr', alpha=0.3)
    # plotting the original data points on top of the decision boundary
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')
    plt.title(title)
    plt.show()


# we give the train and test features and targets to this function
def train_and_evaluate(X_train, X_test, y_train, y_test):
    # we have 3 models, basic linear support vector classifier, polynomial kernel with degree 3, and rbf kernel
    models = {
        "Linear SVM": SVC(kernel='linear', C=1.0),
        "Polynomial SVM (degree=3)": SVC(kernel='poly', degree=3, C=1.0),
        "RBF SVM": SVC(kernel='rbf', C=1.0)
    }
    # to store results of the training process
    results = {}
    for name, model in models.items():
        # feeding the model with the training data
        model.fit(X_train, y_train)
        # predicting both train and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        # calculating accuracy for both sets
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        # since the error rates are mentiones in the assignment, i calculated them here
        # simply subtracting accuracy from 1 gives the error rate
        results[name] = (1 - train_acc, 1 - test_acc)
        #printing both accuracy and error rates
        print(f"{name}:")
        print(f"  → Training error: {1 - train_acc:.3f}")
        print(f"  → Train accuracy:  {train_acc:.3f}")
        print(f"  → Test error:     {1 - test_acc:.3f}\n")
        print(f"  → Test accuracy:  {test_acc:.3f}\n")

        # plotting the decision boundary for each model using the function defined above
        plot_decision_boundary(model, X_train, y_train, f"{name} - Decision Boundary")
    # returning the error rates for summary in main function
    return results


def main():
    # generating and plotting the dataset using the function defined above
    X, y = generate_dataset()
    # splitting the dataset into training and testing sets, 70% train and 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # training and evaluating the models using the function defined above
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    # printing a summary of error rates for all models
    print("Summary of error rates:")
    for model, (train_err, test_err) in results.items():
        print(f"{model:30s} | Train error: {train_err:.3f} | Test error: {test_err:.3f}")


if __name__ == "__main__":
    main()

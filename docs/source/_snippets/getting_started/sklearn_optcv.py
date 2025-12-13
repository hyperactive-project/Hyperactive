"""Scikit-learn OptCV wrapper example for documentation.

This snippet demonstrates how to use OptCV as a drop-in replacement
for GridSearchCV. It is included in get_started.rst.
"""

# [start:full_example]
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from hyperactive.integrations.sklearn import OptCV
from hyperactive.opt.gfo import HillClimbing

# Load and split data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define optimizer with search space
search_space = {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10, 100]}
optimizer = HillClimbing(search_space=search_space, n_iter=20)

# Create tuned estimator (like GridSearchCV)
tuned_svc = OptCV(SVC(), optimizer)

# Fit and predict as usual
tuned_svc.fit(X_train, y_train)
y_pred = tuned_svc.predict(X_test)

# Access results
print(f"Best params: {tuned_svc.best_params_}")
print(f"Best estimator: {tuned_svc.best_estimator_}")
# [end:full_example]

if __name__ == "__main__":
    # Verify we got valid results
    assert hasattr(tuned_svc, "best_params_")
    assert hasattr(tuned_svc, "best_estimator_")
    assert "kernel" in tuned_svc.best_params_
    assert "C" in tuned_svc.best_params_
    print("Sklearn OptCV example passed!")

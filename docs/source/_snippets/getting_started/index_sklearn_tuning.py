"""Scikit-learn tuning example for index page.

This snippet demonstrates sklearn integration using OptCV
shown on the landing page. It is included in index.rst.
"""

# [start:full_example]
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from hyperactive.integrations.sklearn import OptCV
from hyperactive.opt.gfo import HillClimbing

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Define optimizer with search space
search_space = {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10]}
optimizer = HillClimbing(search_space=search_space, n_iter=20)

# Create tuned estimator and fit
tuned_svc = OptCV(SVC(), optimizer)
tuned_svc.fit(X_train, y_train)

print(f"Best params: {tuned_svc.best_params_}")
# [end:full_example]

if __name__ == "__main__":
    # Verify we got valid results
    assert hasattr(tuned_svc, "best_params_")
    assert "kernel" in tuned_svc.best_params_
    assert "C" in tuned_svc.best_params_
    print("Index sklearn tuning example passed!")

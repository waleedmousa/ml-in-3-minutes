# ML in 3 Minutes 09: overfitting, caught with a train/test split
# pip install scikit-learn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
X, y = make_classification(n_samples=2000, n_features=20, n_informative=5,
                           flip_y=0.1, random_state=0)   # made-up "transactions"
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
for depth in [None, 4]:                                  # no limit vs simple
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, y_tr)
    print(f"max_depth={depth}: train {tree.score(X_tr, y_tr):.1%}  test {tree.score(X_te, y_te):.1%}")

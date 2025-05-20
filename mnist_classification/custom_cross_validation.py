from sklearn import clone
from sklearn.model_selection import StratifiedKFold


def custom_cross_val_score(x_train, y_train_5, model, scoring="accuracy"):
    skfolds = StratifiedKFold(n_splits=3)
    for train_index, test_index in skfolds.split(x_train, y_train_5):
        clone_clf = clone(model)
        x_train_folds = x_train[train_index]
        y_train_folds = y_train_5[train_index]
        x_test_fold = x_train[test_index]
        y_test_fold = y_train_5[test_index]

        clone_clf.fit(x_train_folds, y_train_folds)
        y_pred = clone_clf.predict(x_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(f"Correct predictions: {n_correct} out of {len(y_test_fold)}")
        print(f"Accuracy: {n_correct / len(y_test_fold)}")

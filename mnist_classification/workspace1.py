from matplotlib import pyplot as plt
from sklearn.calibration import cross_val_predict
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score

from plot_digit import plot_digit

mnist = fetch_openml("mnist_784", as_frame=False)

X, y = mnist["data"], mnist["target"]

# some_digit = X[0]
# plot_digit(some_digit)
# plt.show()

x_train, x_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# for i in range(100):
#    plt.subplot(10, 10, i + 1)
#    plot_digit(x_train[i])
#
# plt.show()

y_train_5 = y_train == "5"
y_test_5 = y_test == "5"

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train_5)


# for i in range(9):
#    plt.subplot(3, 3, i + 1)
#    value = sgd_clf.predict([x_train[i]])
#    plt.title(f"{value[0]}")
#    plot_digit(x_train[i])
#
# plt.show()
# print("SGDClassifier score: ", sgd_clf.score(x_test, y_test_5))
# print(
#    "Cross-validation scores: ",
#    cross_val_score(sgd_clf, x_train, y_train_5, cv=3, scoring="accuracy"),
# )
#
# dummy_clf = DummyClassifier(strategy="most_frequent")
# dummy_clf.fit(x_train, y_train_5)
# print("DummyClassifier score: ", dummy_clf.score(x_test, y_test_5))

# y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3)
# cm = confusion_matrix(y_train_5, y_train_pred)
# print("Confusion matrix:\n", cm)
#
# print("Precision:", precision_score(y_train_5, y_train_pred))
# print("Recall:", recall_score(y_train_5, y_train_pred))
# print("F1 score:", f1_score(y_train_5, y_train_pred))
#
# y_scores = sgd_clf.decision_function(x_train[0:1])
# print("Decision function score:", y_scores)

y_scores = cross_val_predict(
    sgd_clf, x_train, y_train_5, cv=3, method="decision_function"
)

# print("y_scores shape:", y_scores.shape)
# print("y_scores[:10]:", y_scores[:10])
# print("y_scores[-10:]:", y_scores[-10:])

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
#
# plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
# plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
# plt.vlines(thresholds, 0, 1, "k", linestyles="dotted", label="Threshold")
## print grid, legend, axis labels and circle
# plt.legend(loc="upper right", fontsize=12)
# plt.xlabel("Threshold", fontsize=12)
# plt.ylabel("Score", fontsize=12)
# plt.title("Precision and Recall vs Threshold", fontsize=14)
# plt.axis([-50000, 50000, 0, 1])
# plt.grid()
# plt.show()
#

# plt.plot(recalls, precisions, "b-", linewidth=2, label="Precision vs Recall")
# plt.xlabel("Recall", fontsize=12)
# plt.ylabel("Precision", fontsize=12)
# plt.title("Precision vs Recall", fontsize=14)
# plt.grid()
# plt.legend(loc="upper right", fontsize=12)
# plt.show()

idx_90_precision = (
    precisions >= 0.90
).argmax()  # This is the index of the first precision value that is greater than or equal to 0.90
thresholds_90_precision = thresholds[
    idx_90_precision
]  # This is the threshold value that gives us 90% precision
# print("Threshold for 90% precision:", thresholds_90_precision)
#
# y_train_pred_90_precision = y_scores >= thresholds_90_precision
# print("Precision:", precision_score(y_train_5, y_train_pred_90_precision))
# print("Recall:", recall_score(y_train_5, y_train_pred_90_precision))
# print("F1 score:", f1_score(y_train_5, y_train_pred_90_precision))


fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
idx_for_threshold_at_90 = (
    thresholds <= thresholds_90_precision
).argmax()  # This is the index of the first threshold value that is greater than or equal to the threshold for 90% precision
tpr_at_90 = tpr[
    idx_for_threshold_at_90
]  # This is the true positive rate at that threshold
fpr_at_90 = fpr[
    idx_for_threshold_at_90
]  # This is the false positive rate at that threshold

# plt.plot(fpr, tpr, "b-", linewidth=2, label="ROC curve")
# plt.grid()
# plt.xlabel("False Positive Rate (Fallout)", fontsize=12)
# plt.ylabel("True Positive Rate (Recall)", fontsize=12)
# plt.plot(
#    [0, 1], [0, 1], "k--", label="Random", linewidth=2
# )  # This is the diagonal line representing random guessing
# plt.plot([fpr_at_90], [tpr_at_90], "ro", label="Threshold for 90% precision")
# plt.title("ROC Curve", fontsize=14)
# plt.legend(loc="lower right", fontsize=12)
# plt.show()

roc_auc = roc_auc_score(y_train_5, y_scores)
print("ROC AUC score:", roc_auc)

forent_clf = RandomForestClassifier(random_state=42)
y_probs_forest = cross_val_predict(
    forent_clf, x_train, y_train_5, cv=3, method="predict_proba"
)
print("y_probs_forest shape:", y_probs_forest[:2])
precision_forest, recall_forest, thresholds_forest = precision_recall_curve(
    y_train_5, y_probs_forest[:, 1]
)

plt.plot(recall_forest, precision_forest, "b-", linewidth=2, label="Random Forest")
plt.plot(recalls, precisions, "g-", linewidth=2, label="SGD")
plt.grid()
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.show()

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


mnist = fetch_openml("mnist_784", as_frame=False)

X, y = mnist["data"], mnist["target"]
x_train, x_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

svc_clf = SVC(random_state=42)
# svc_clf.fit(x_train[:2000], y_train[:2000])

some_digit = x_train[0]
# svc_pred = svc_clf.predict([some_digit])
# print("SVC prediction: ", svc_pred)
#
# some_digit_score = svc_clf.decision_function([some_digit])
# print("SVC decision function score: ", some_digit_score)
#
# class_id = some_digit_score.argmax()
# print("Class ID: ", class_id)
# print("Class name: ", svc_clf.classes_)
# print("Predicted class: ", svc_clf.classes_[class_id])


# ovr_clf = OneVsRestClassifier(SVC(random_state=42))
# ovr_clf.fit(x_train[:2000], y_train[:2000])
# ovr_pred = ovr_clf.predict([some_digit])
# print("OneVsRestClassifier prediction: ", ovr_pred)
# print(len(ovr_clf.estimators_))

sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(x_train, y_train)
# sgd_pred = sgd_clf.predict([some_digit])
# print("SGDClassifier prediction: ", sgd_pred)
#
# dec_func = sgd_clf.decision_function([some_digit]).round(2)
# print("SGDClassifier decision function score: ", dec_func)
#
# cross_val_scores = cross_val_score(sgd_clf, x_train, y_train, cv=3, scoring="accuracy")
# print("Cross-validation scores: ", cross_val_scores)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float64))
cross_val_scores_scaled = cross_val_score(
    sgd_clf, x_train_scaled, y_train, cv=3, scoring="accuracy"
)
print("Cross-validation scores with scaling: ", cross_val_scores_scaled)

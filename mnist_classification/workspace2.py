from sklearn.datasets import fetch_openml
from sklearn.svm import SVC


mnist = fetch_openml("mnist_784", as_frame=False)

X, y = mnist["data"], mnist["target"]
x_train, x_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

svc_clf = SVC(random_state=42)
svc_clf.fit(x_train[:2000], y_train[:2000])

some_digit = x_train[4]
svc_pred = svc_clf.predict([some_digit])
print("SVC prediction: ", svc_pred)

some_digit_score = svc_clf.decision_function([some_digit])
print("SVC decision function score: ", some_digit_score)

class_id = some_digit_score.argmax()
print("Class ID: ", class_id)

print("Predicted class: ", svc_clf.classes_[class_id])

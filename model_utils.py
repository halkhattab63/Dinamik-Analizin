import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize

def train_knn(X_train, y_train, n_neighbors=5, metric='minkowski', p=2):
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, p=p)
    classifier.fit(X_train, y_train)
    return classifier
def evaluate_model(classifier, X_test, y_test, y_binned):
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)  # حل التحذير
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # حساب ROC و AUC
    y_test_bin = label_binarize(y_test, classes=np.unique(y_binned))
    n_classes = y_test_bin.shape[1]

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        if np.sum(y_test_bin[:, i]) > 0:
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], (y_pred == i).astype(int))
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            fpr[i], tpr[i], roc_auc[i] = [0], [0], 0

    return cm, accuracy, precision, recall, f1, fpr, tpr, roc_auc

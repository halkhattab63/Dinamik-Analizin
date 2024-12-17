import cProfile
import tracemalloc
from data_utils import load_dataset, preprocess_data
from model_utils import train_knn, evaluate_model
from visualization_utils import plot_confusion_matrix, plot_roc_curve, visualize_results
import numpy as np

tracemalloc.start()

def main():
    dataset_path = 'winequality-red.csv'

    # تحميل البيانات
    X, y = load_dataset(dataset_path)

    # معالجة البيانات
    X_train, X_test, y_train, y_test, y_binned = preprocess_data(X, y)

    # تدريب النموذج
    classifier = train_knn(X_train, y_train)

    # تقييم النموذج
    cm, accuracy, precision, recall, f1, fpr, tpr, roc_auc = evaluate_model(classifier, X_test, y_test, y_binned)

    # طباعة النتائج
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # الرسوم البيانية
    plot_confusion_matrix(cm)
    plot_roc_curve(fpr, tpr, roc_auc, num_classes=len(np.unique(y_binned)))
    visualize_results(X_train, y_train, classifier, 'KNN (Training Set)')
    visualize_results(X_test, y_test, classifier, 'KNN (Test Set)')

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 5 Memory Consumers ]")
    for stat in top_stats[:5]:
        print(stat)

if __name__ == "__main__":
    cProfile.run("main()")

import numpy as np

class Evaluator:
    @staticmethod
    def calculate_confusion_matrix(predictions, y_test):
        unique_labels = np.unique(np.concatenate([y_test, predictions]))
        num_labels = len(unique_labels)
        confusion_matrix = np.zeros((num_labels, num_labels), dtype=int)

        for true_label, pred_label in zip(y_test, predictions):
            row = np.where(unique_labels == true_label)[0][0]
            col = np.where(unique_labels == pred_label)[0][0]
            confusion_matrix[row, col] += 1

        return confusion_matrix

    @staticmethod
    def calculate_accuracy(predictions, y_test):
        correct_predictions = np.sum(predictions == y_test)
        total_predictions = len(y_test)
        accuracy = correct_predictions / total_predictions
        return accuracy

    @staticmethod
    def calculate_precision(predictions, y_test):
        unique_labels = np.unique(np.concatenate([y_test, predictions]))
        num_labels = len(unique_labels)
        precision_per_class = np.zeros(num_labels, dtype=float)

        for label in unique_labels:
            true_positive = np.sum((predictions == label) & (y_test == label))
            false_positive = np.sum((predictions == label) & (y_test != label))
            precision_per_class[label] = true_positive / (true_positive + false_positive) if true_positive + false_positive != 0 else 0

        precision = np.mean(precision_per_class)
        return precision

    @staticmethod
    def calculate_recall(predictions, y_test):
        unique_labels = np.unique(np.concatenate([y_test, predictions]))
        num_labels = len(unique_labels)
        recall_per_class = np.zeros(num_labels, dtype=float)

        for label in unique_labels:
            true_positive = np.sum((predictions == label) & (y_test == label))
            false_negative = np.sum((predictions != label) & (y_test == label))
            recall_per_class[label] = true_positive / (true_positive + false_negative) if true_positive + false_negative != 0 else 0

        recall = np.mean(recall_per_class)
        return recall

    @staticmethod
    def calculate_f1_score(predictions, y_test):
        precision = Evaluator.calculate_precision(predictions, y_test)
        recall = Evaluator.calculate_recall(predictions, y_test)
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
        return f1

    @staticmethod
    def evaluate_model(predictions, y_test):
        cm = Evaluator.calculate_confusion_matrix(predictions, y_test)
        print("Confusion Matrix:")
        print(cm)

        accuracy = Evaluator.calculate_accuracy(predictions, y_test)
        print(f"\nAccuracy: {accuracy:.4f}")

        precision = Evaluator.calculate_precision(predictions, y_test)
        print(f"Precision: {precision:.4f}")

        recall = Evaluator.calculate_recall(predictions, y_test)
        print(f"Recall: {recall:.4f}")

        f1 = Evaluator.calculate_f1_score(predictions, y_test)
        print(f"F1 Score: {f1:.4f}\n")


class ClusterEvaluator:
    @staticmethod
    def clusters_accuracy(k_labels, y_train, k_value, i, j):
        mask_i = (k_labels == i)
        mask_j = (k_labels == j)
        mask_ij = mask_i | mask_j

        true_labels_ij = np.where(mask_i, y_train, np.where(mask_j, y_train, -1))
        predicted_labels_ij = np.where(mask_i, i, np.where(mask_j, j, -1))

        accuracy_ij = Evaluator.calculate_accuracy(true_labels_ij[mask_ij], predicted_labels_ij[mask_ij])

        return accuracy_ij * 100
    
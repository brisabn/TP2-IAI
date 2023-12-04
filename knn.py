import numpy as np
import matplotlib.pyplot as plt

class KNN:
    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    @staticmethod
    def k_nearest_neighbors(x_train, y_train, x_test, k):
        predictions = []
        for test_sample in x_test:
            distances = [(KNN.euclidean_distance(test_sample, train_sample), label) for train_sample, label in zip(x_train, y_train)]
            sorted_distances = sorted(distances, key=lambda x: x[0])[:k]
            k_nearest_labels = [label for distance, label in sorted_distances]
            predicted_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(predicted_label)
        return predictions

    @staticmethod
    def plot_3d_scatter(attributes, predictions, y_test, k_value, attribute_names, accuracy):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors_and_labels = {
            (0, True): ('darkblue', 'Class 0 - Correct'),
            (0, False): ('royalblue', 'Class 0 - Incorrect'),
            (1, True): ('darkred', 'Class 1 - Correct'),
            (1, False): ('indianred', 'Class 1 - Incorrect'),
        }

        for class_label, correct in colors_and_labels:
            mask = (np.array(predictions) == class_label) & (np.array(predictions) == y_test) if correct else \
                (np.array(predictions) == class_label) & (np.array(predictions) != y_test)

            ax.scatter(attributes[mask][:, 0], attributes[mask][:, 1], attributes[mask][:, 2],
                    c=colors_and_labels[(class_label, correct)][0],
                    label=colors_and_labels[(class_label, correct)][1], alpha=0.7)

        ax.set_title(f'KNN Predictions with k = {k_value}')
        ax.text2D(0.05, 0.95, f'Accuracy: {accuracy * 100:.2f}%', transform=ax.transAxes)

        ax.set_xlabel(attribute_names[0])
        ax.set_ylabel(attribute_names[1])
        ax.set_zlabel(attribute_names[2])

        # Adicione a legenda
        ax.legend()
        plt.show()


    @staticmethod
    def plot_accuracies(k_values, all_accuracies, num_iterations, use_scikit):
        colors = ['tomato', 'hotpink', 'gold', 'lightseagreen']
        color_index = 0

        for i, k_value in enumerate(k_values):
            k_accuracies_for_k = [accuracies[i] for accuracies in all_accuracies]
            plt.plot(range(1, num_iterations + 1), k_accuracies_for_k, marker='o', label=f'k={k_value}', color=colors[color_index])

            color_index = (color_index + 1) % len(colors)  # Cycle through colors

        if use_scikit:
            plt.title("KNN Accuracy (Multiple Iterations) - Scikit")
        else:
            plt.title("KNN Accuracy for Different Values of k (Multiple Iterations)")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.show()
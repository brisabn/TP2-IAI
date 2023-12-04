import numpy as np
import matplotlib.pyplot as plt

class K_Means:
    @staticmethod
    def k_means(X, k, max_iters=4000):
        centroids = X[np.random.choice(X.shape[0], k, replace=False)]

        for _ in range(max_iters):
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

            if np.all(centroids == new_centroids):
                break

            centroids = new_centroids

        return labels, centroids

    @staticmethod
    def plot_3d_scatter(attributes, labels, centroids, attribute_names, k_value, y_true, accuracy=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        unique_labels = np.unique(labels)
        cluster_colors = ['darkblue', 'darkred', 'lightsteelblue'][:k_value]

        for i, label in enumerate(unique_labels):
            cluster_mask = (labels == label)

            # Assign colors based on k_value
            if k_value == 3:
                ax.scatter(attributes[cluster_mask, 0], attributes[cluster_mask, 1], attributes[cluster_mask, 2], c=cluster_colors[i], marker='o', alpha=0.5, label=f'Cluster {label}')
            elif k_value == 2:
                correct_mask = (labels == y_true)
                incorrect_mask = ~correct_mask
                correct_cluster_mask = correct_mask & (labels == label)
                incorrect_cluster_mask = incorrect_mask & (labels == label)

                if label == 0:
                    ax.scatter(attributes[correct_cluster_mask, 0], attributes[correct_cluster_mask, 1], attributes[correct_cluster_mask, 2], c=cluster_colors[i], marker='o', alpha=0.7, label=f'Cluster {label} - Correct')
                    ax.scatter(attributes[incorrect_cluster_mask, 0], attributes[incorrect_cluster_mask, 1], attributes[incorrect_cluster_mask, 2], c='steelblue', marker='o', alpha=0.7, label=f'Cluster {label} - Incorrect')
                elif label == 1:
                    ax.scatter(attributes[correct_cluster_mask, 0], attributes[correct_cluster_mask, 1], attributes[correct_cluster_mask, 2], c=cluster_colors[i], marker='o', alpha=0.7, label=f'Cluster {label} - Correct')
                    ax.scatter(attributes[incorrect_cluster_mask, 0], attributes[incorrect_cluster_mask, 1], attributes[incorrect_cluster_mask, 2], c='palevioletred', marker='o', alpha=0.7, label=f'Cluster {label} - Incorrect')

        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', marker='*', s=150, alpha=1.0, label='Centroids')

        ax.set_xlabel(attribute_names[0])
        ax.set_ylabel(attribute_names[1])
        ax.set_zlabel(attribute_names[2])

        # Evaluate the model and plot accuracy
        if k_value == 2:
            plt.title(f'KMeans Clustering (k={k_value})')
            ax.text2D(0.05, 0.95, f'Accuracy: {accuracy:.2f}%', transform=ax.transAxes)
        else:
            plt.title(f'KMeans Clustering (k={k_value})')

        ax.legend()  # Add legend
        plt.show()

    @staticmethod
    def plot_accuracies(k_values, all_accuracies, num_iterations, use_scikit):
        colors = ['gold', 'mediumvioletred', 'lightseagreen', 'tomato']  # Customize colors here
        color_index = 0

        for k_value in k_values:
            plt.figure()
            for (i, j), acc_list in all_accuracies[k_value].items():
                # Plot accuracy values
                plt.plot(range(1, num_iterations + 1), acc_list[:num_iterations], marker='o', label=f'Cluster {i} and Cluster {j}, k = {k_value}', color=colors[color_index], linewidth=1, markersize=5)

                color_index = (color_index + 1) % len(colors)  # Cycle through colors

            # Calculate and plot the mean accuracy line
            mean_accuracies = np.mean([acc_list[:num_iterations] for acc_list in all_accuracies[k_value].values()], axis=0)
            plt.axhline(y=np.mean(mean_accuracies), linestyle='--', color='black', label='Mean Accuracy', linewidth=2)

            if use_scikit:
                plt.title(f'Accuracies for k={k_value} (Multiple Iterations) - Scikit')
            else:
                plt.title(f'Accuracies for k={k_value} (Multiple Iterations)')
            plt.xlabel('Iteration')
            plt.ylabel('Accuracy (%)')
            plt.legend(fontsize='small')
            plt.show()

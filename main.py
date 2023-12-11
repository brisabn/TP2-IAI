import sys
from evaluation import Evaluator, ClusterEvaluator
from data import load_and_split_data, get_attribute_names
from knn import KNN
from kmeans import K_Means

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

if len(sys.argv) not in [4, 5, 6]:
    print("Usage: python TP2.py <training_file> <testing_file> <algorithm> [y | scatter] [scikit]")
    sys.exit(1)

# Command-line arguments
training_file = sys.argv[1]
testing_file = sys.argv[2]
algorithm = sys.argv[3].lower()
target_column = "TARGET_5Yrs"

# Check if "y", "scatter", or "scikit" is present
plot_accuracies = False
plot_scatter = False
use_scikit = False
if len(sys.argv) >= 5:
    if sys.argv[4].lower() == 'y':
        plot_accuracies = True
    elif sys.argv[4].lower() == 'scatter':
        plot_scatter = True
    if sys.argv[4].lower() == 'scikit' or (len(sys.argv) > 5 and sys.argv[5].lower() == 'scikit'):
        use_scikit = True

# Attribute names to print later
attribute_names = get_attribute_names(training_file)

# Executar KNN
def run_knn(num_iterations, plot_accuracies, plot_scatter, use_scikit):
    k_values = [2, 10, 50, 100]
    all_accuracies = []

    for _ in range(num_iterations):
        x_train, y_train, x_test, y_test = load_and_split_data(training_file, testing_file, target_column)
        k_accuracies = []

        for k_value in k_values:
            if not use_scikit: # KNN Implementado
                predictions = KNN.k_nearest_neighbors(x_train, y_train, x_test, k_value)
            else: # KNN do Scikit-learn
                if not plot_accuracies:
                    print(f"KNN with k = {k_value} from Scikit-Learn - Comparative Purposes")
                knn_classifier = KNeighborsClassifier(n_neighbors=k_value)
                knn_classifier.fit(x_train, y_train)
                predictions = knn_classifier.predict(x_test)

            # Accuracy
            accuracy = Evaluator.calculate_accuracy(y_test, predictions) * 100
            k_accuracies.append(accuracy)

            # Print Metrics
            if not plot_accuracies:
                print(f"Metrics for KNN with k = {k_value}")
                Evaluator.evaluate_model(predictions, y_test)

            # Plot 3D scatter
            if plot_scatter:
                condensed_attributes = x_test[:, [0, 2, 11]]
                KNN.plot_3d_scatter(condensed_attributes, predictions, y_test, k_value, attribute_names, accuracy)

        # accuracies for each k value in the iteration
        all_accuracies.append(k_accuracies)

    # Multiple Test
    if plot_accuracies:
        KNN.plot_accuracies(k_values, all_accuracies, num_iterations, use_scikit)

# Executar KMeans
def run_kmeans(num_iterations, plot_accuracies, plot_scatter, use_scikit):
    k_values = [2, 3]
    all_accuracies = {k: {pair: [] for pair in [(i, j) for i in range(k) for j in range(i + 1, k)]} for k in k_values}

    for _ in range(num_iterations):
        x_train, y_train, x_test, y_test = load_and_split_data(training_file, testing_file, target_column)

        for k_value in k_values:

            if not use_scikit: # KMeans Implementado
                k_labels, k_centroids = K_Means.k_means(x_train, k_value)
            else: # KMeans do Scikit-learn
                if not plot_accuracies:
                    print(f"KMeans with k = {k_value} from Scikit-Learn - Comparative Purposes")
                kmeans = KMeans(n_clusters=k_value, n_init=k_value*5)
                k_labels = kmeans.fit_predict(x_train)
                k_centroids = kmeans.cluster_centers_

            # Accuracy for pairs of clusters
            for i, j in [(i, j) for i in range(k_value) for j in range(i + 1, k_value)]:
                accuracy_ij = ClusterEvaluator.clusters_accuracy(k_labels, y_train, k_value, i, j)
                all_accuracies[k_value][(i, j)].append(accuracy_ij)

                if not plot_accuracies:
                    print(f"Accuracy for Cluster {i} and Cluster {j} with k={k_value}: {accuracy_ij:.2f}%")

            # Centroids of clusters
            if not plot_accuracies:
                print(f"Centroids for k={k_value}:")
                print(k_centroids)
                print()

            # Plot 3D scatter
            if plot_scatter:
                condensed_attributes = x_train[:, [0, 2, 11]]
                K_Means.plot_3d_scatter(condensed_attributes, k_labels, k_centroids, attribute_names, k_value, y_train, accuracy_ij)

    # Multiple Test
    if plot_accuracies:
        K_Means.plot_accuracies(k_values, all_accuracies, num_iterations, use_scikit)

# Algorithm function dictionary
algorithm_functions = {
    'knn': run_knn,
    'kmeans': run_kmeans,
}

# Algorithm run
multiple_iterations = 30
if algorithm in algorithm_functions:
    if plot_accuracies:
        algorithm_functions[algorithm](multiple_iterations, True, False, use_scikit)
    else:
        algorithm_functions[algorithm](1, False, plot_scatter, use_scikit)
else:
    print("Invalid algorithm. Please choose either 'knn' or 'kmeans'.")
    sys.exit(1)

import numpy as np

class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None        

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """

        converged = False

        # Initialize the means (centroids) by randomy choosing 4 samples from the dataset
        features_copy = features.copy()
        np.random.shuffle(features_copy)
        self.means = features_copy[:self.n_clusters]

        # Intialize an ID array of length M to cluster samples
        old_centroids = np.zeros(features.shape[0]).astype(int)

        while not converged:
            new_centroids = []
            # Step 1: Update assignments based on current self.means
            for r in range(features.shape[0]):
                distances = np.array([
                    np.sum(np.square(features[r] - self.means[i]))
                    for i in range(len(self.means))
                ])
                new_centroids.append(np.argmin(distances))
            new_centroids = np.array(new_centroids)

            # Step 2: Update means (centroids) based on the recently classified clusters
            self.means = np.array([
                features[new_centroids == i].mean(axis = 0)
                for i in range(self.n_clusters)
            ])

            # Step 3: Check converged status
            converged = (old_centroids == np.array(new_centroids)).all()
            old_centroids = new_centroids

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        predictions = []
        
        for r in range(features.shape[0]):
            distances = np.array([
                np.sum(np.square(features[r] - self.means[i]))
                for i in range(len(self.means))
            ])
            predictions.append(np.argmin(distances, axis = 0))

        return np.array(predictions)
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel


# This class computes the similarity of each sample to the cluster centers
# using the RBF kernel. It is a custom transformer that can be used in a pipeline.
# It uses KMeans clustering to find the cluster centers and then computes
# the RBF kernel similarity between the samples and the cluster centers.
# The RBF works by computing the similarity between each sample and the cluster centers
# using the formula: K(x, y) = exp(-gamma * ||x - y||^2)
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=8, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(
            X,
            self.kmeans_.cluster_centers_,
            gamma=self.gamma,
        )

    def get_feature_names_out(self, input_features=None):
        return [f"cluster_similarity_{i}" for i in range(self.n_clusters)]

"""
Implementing and Interpreting Principal Component Analysis (PCA) from Scratch
----------------------------------------------------------------------------

Steps covered:

1. Generate a synthetic dataset with at least 10 features and 3 classes
   using sklearn.datasets.make_classification.

2. Implement PCA from scratch using NumPy:
   - Data centering
   - Covariance matrix
   - Eigen decomposition
   - Sorting eigenvalues/eigenvectors
   - Explained variance & explained variance ratio

3. Determine optimal number of components K using cumulative explained
   variance ratio and plotting it.

4. Project the dataset onto the top K principal components and:
   - Show first 10 rows of transformed data
   - Plot first two PCs colored by class labels
   - Interpret PC1 and PC2 in terms of original features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


# -------------------------------------------------------------------
# 1. Generate synthetic dataset (10+ features, 3 distinct classes)
# -------------------------------------------------------------------

def generate_dataset(
    n_samples=1000,
    n_features=10,
    n_informative=7,
    n_redundant=2,
    n_classes=3,
    random_state=42
):
    """
    Generate a synthetic classification dataset with non-trivial covariance.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=1.5,
        flip_y=0.01,
        random_state=random_state
    )

    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="class")

    return X_df, y_series


# -------------------------------------------------------------------
# 2. Implement PCA from scratch using NumPy
# -------------------------------------------------------------------

class CustomPCA:
    """
    PCA implementation from scratch using NumPy.

    Attributes after fitting:
    - mean_: mean of each feature
    - components_: eigenvectors (principal axes), shape (n_features, n_components)
    - explained_variance_: eigenvalues of covariance matrix
    - explained_variance_ratio_: proportion of variance explained by each component
    - cumulative_explained_variance_ratio_: cumulative sum of explained variance ratio
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.cumulative_explained_variance_ratio_ = None

    def fit(self, X: np.ndarray):
        """
        Fit PCA model on data X (numpy array of shape (n_samples, n_features)).
        """
        # 1) Center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2) Covariance matrix (features on columns => rowvar=False)
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3) Eigen decomposition
        # Covariance matrix is symmetric -> use eigh (guaranteed real eigenvalues)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)

        # 4) Sort eigenvalues and eigenvectors in descending order
        sorted_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_idx]
        eigvecs = eigvecs[:, sorted_idx]

        # 5) Store explained variance stats
        self.explained_variance_ = eigvals
        total_var = np.sum(eigvals)
        self.explained_variance_ratio_ = eigvals / total_var
        self.cumulative_explained_variance_ratio_ = np.cumsum(
            self.explained_variance_ratio_
        )

        # 6) Select top n_components eigenvectors as principal axes
        if self.n_components is None:
            self.components_ = eigvecs  # all components
        else:
            self.components_ = eigvecs[:, : self.n_components]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data X onto the principal components.
        """
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("You must fit the PCA before calling transform().")

        X_centered = X - self.mean_
        # Note: components_ shape is (n_features, n_components)
        return np.dot(X_centered, self.components_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Convenience method: fit then transform.
        """
        self.fit(X)
        return self.transform(X)


# -------------------------------------------------------------------
# 3. Determine optimal number of principal components (K)
# -------------------------------------------------------------------

def choose_optimal_k(cumulative_explained_variance_ratio, threshold=0.95):
    """
    Choose the minimum K such that cumulative explained variance >= threshold.
    """
    return int(np.argmax(cumulative_explained_variance_ratio >= threshold) + 1)


def plot_cumulative_explained_variance(cum_var_ratio):
    """
    Plot cumulative explained variance ratio vs number of components.
    """
    n_components = len(cum_var_ratio)
    plt.figure()
    plt.plot(range(1, n_components + 1), cum_var_ratio, marker="o")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("Cumulative Explained Variance vs Number of Components")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 4. Visualization of projected data (first two PCs)
# -------------------------------------------------------------------

def plot_first_two_pcs(X_pca_2d, y):
    """
    Scatter plot of first two principal components, colored by class label.
    """
    plt.figure()
    for class_label in np.unique(y):
        mask = y == class_label
        plt.scatter(
            X_pca_2d[mask, 0],
            X_pca_2d[mask, 1],
            label=f"Class {class_label}",
            alpha=0.7,
        )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Projection onto First Two Principal Components")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 5. Interpretation helpers
# -------------------------------------------------------------------

def interpret_first_two_pcs(components, feature_names, top_n=3):
    """
    Print interpretation of PC1 and PC2 in terms of original features.

    We look at the loadings (eigenvector coefficients) and pick features with
    the largest absolute values (strongest contribution) for each PC.
    """
    pc1 = components[:, 0]
    pc2 = components[:, 1]

    # Get indices sorted by absolute loading magnitude
    pc1_idx = np.argsort(np.abs(pc1))[::-1][:top_n]
    pc2_idx = np.argsort(np.abs(pc2))[::-1][:top_n]

    print("\n--- Interpretation of Principal Components ---")
    print("PC1 is most strongly influenced by:")
    for idx in pc1_idx:
        print(f"  {feature_names[idx]} (loading = {pc1[idx]:.4f})")

    print("\nPC2 is most strongly influenced by:")
    for idx in pc2_idx:
        print(f"  {feature_names[idx]} (loading = {pc2[idx]:.4f})")


# -------------------------------------------------------------------
# 6. Main execution
# -------------------------------------------------------------------

def main():
    # Step 1: Generate dataset
    X_df, y = generate_dataset()
    feature_names = X_df.columns.tolist()
    X = X_df.values

    print("Shape of original dataset:", X.shape)  # (n_samples, 10 features)

    # Step 2: Fit PCA with all components first (to analyze variance)
    pca_full = CustomPCA(n_components=None)
    pca_full.fit(X)

    print("\nEigenvalues (Explained Variance) for each component:")
    print(pca_full.explained_variance_)

    print("\nExplained variance ratio for each component:")
    print(pca_full.explained_variance_ratio_)

    print("\nCumulative explained variance ratio:")
    print(pca_full.cumulative_explained_variance_ratio_)

    # Step 3: Plot cumulative explained variance ratio
    plot_cumulative_explained_variance(pca_full.cumulative_explained_variance_ratio_)

    # Choose K based on 95% variance threshold (you can change threshold if needed)
    threshold = 0.95
    K = choose_optimal_k(pca_full.cumulative_explained_variance_ratio_, threshold)

    print(f"\nChosen number of components K = {K} "
          f"to retain at least {threshold*100:.1f}% of total variance.")

    # Step 4: Fit PCA again with K components and transform data
    pca_k = CustomPCA(n_components=K)
    X_pca_k = pca_k.fit_transform(X)

    # Store transformed data in DataFrame for easy viewing
    pc_columns = [f"PC{i+1}" for i in range(K)]
    X_pca_df = pd.DataFrame(X_pca_k, columns=pc_columns)

    print("\nFirst 10 rows of the transformed (reduced-dimension) dataset:")
    print(X_pca_df.head(10))

    # Step 5: Visualization using first two principal components
    # Even if K < 2 (unlikely for 95% threshold), we only plot if we have 2 PCs
    if K >= 2:
        X_pca_2d = X_pca_k[:, :2]
        plot_first_two_pcs(X_pca_2d, y.values)
        interpret_first_two_pcs(pca_k.components_, feature_names, top_n=3)
    else:
        print("\nK < 2, so we cannot plot first two principal components.")

    # Optional: save transformed data to CSV if needed
    # X_pca_df.to_csv("transformed_dataset.csv", index=False)


if __name__ == "__main__":
    main()

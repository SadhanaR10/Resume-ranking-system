import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Function to compute cosine similarity between sparse matrices X and Y
def cosine_similarity_sparse(X, Y):
    dot_product = X.dot(Y.T).toarray()
    norm_X = np.sqrt((X.multiply(X)).sum(axis=1).A1)
    norm_Y = np.sqrt((Y.multiply(Y)).sum(axis=1).A1)
    similarity = dot_product / (norm_X[:, np.newaxis] * norm_Y)
    return similarity
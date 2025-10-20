# Query_Your_Model/tests/test_similarity.py
import numpy as np
from Query_Your_Model.core.retrieval import combined_similarity


def test_combined_similarity_basic():
    a = np.array([1, 0, 0], dtype="float32")   # feature vector 1
    b = np.array([0, 1, 0], dtype="float32")   # feature vector 2 (orthogonal)
    shap_a = np.array([0.5, 0.2, 0.1], dtype="float32")  # shap for a
    shap_b = np.array([-0.5, 0.0, 0.0], dtype="float32") # shap for b

    # Similarity of identical pair (a,a)
    s1 = combined_similarity(a, shap_a, a, shap_a, alpha=0.5)
    print(f"Similarity (identical): {s1:.4f}")
    assert s1 > 0.99, "Expected similarity close to 1 for identical vectors"

    # Similarity of different pair (a,b)
    s2 = combined_similarity(a, shap_a, b, shap_b, alpha=0.5)
    print(f"Similarity (different/orthogonal): {s2:.4f}")
    assert s2 < s1, "Expected orthogonal similarity to be smaller"

    return s1, s2


if __name__ == "__main__":
    print("Running combined_similarity tests...\n")
    s1, s2 = test_combined_similarity_basic()
    print("\n Test passed!")

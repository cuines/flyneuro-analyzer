"""Analyzer for Drosophila connectome data."""
import numpy as np
import math

def calculate_pathway_entropy(probabilities):
    """
    Calculate Shannon entropy of a discrete probability distribution.
    
    Parameters
    ----------
    probabilities : array-like
        List of probabilities summing to 1.
    
    Returns
    -------
    entropy : float
        Shannon entropy in bits.
    
    Raises
    ------
    ValueError
        If probabilities are not valid.
    """
    # Ensure input is a numpy array
    p = np.asarray(probabilities, dtype=float)
    if not np.allclose(p.sum(), 1.0, atol=1e-9):
        raise ValueError("Probabilities must sum to 1.")
    if np.any(p < 0):
        raise ValueError("Probabilities cannot be negative.")
    
    # BUG: using natural log instead of log base 2
    entropy = -np.sum(p * np.log(p))
    return entropy

def test_entropy():
    """Simple test to demonstrate bug."""
    # Uniform distribution over 4 states: entropy should be 2 bits
    p_uniform = [0.25, 0.25, 0.25, 0.25]
    entropy = calculate_pathway_entropy(p_uniform)
    expected = 2.0  # -4 * 0.25 * log2(0.25) = 2
    print(f"Uniform distribution entropy: {entropy:.4f} (expected {expected})")
    print(f"Difference: {entropy - expected:.4f}")
    return entropy

if __name__ == "__main__":
    test_entropy()
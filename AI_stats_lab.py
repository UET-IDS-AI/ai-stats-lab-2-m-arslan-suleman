"""
AI Mathematical Tools – Probability & Random Variables

Instructions:
- Implement ALL functions.
- Do NOT change function names or signatures.
- Do NOT print inside functions.
- You may use: math, numpy, matplotlib.
"""

import math
from multiprocessing import Value
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Part 1 — Probability Foundations
# ============================================================

def probability_union(PA, PB, PAB):
    """
    P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
    """
    # In the case of the independent events the intersection would be zero
    return PA+PB - (PAB)


def conditional_probability(PAB, PB):
    """
    P(A|B) = P(A ∩ B) / P(B)
    """
    return (PAB)/PB


def are_independent(PA, PB, PAB, tol=1e-9):
    """
    True if:
        |P(A ∩ B) - P(A)P(B)| < tol
    """
    if abs((PAB) - (PA*PB)) < tol:
        return True
    return False



def bayes_rule(PBA, PA, PB):
    """
    P(A|B) = P(B|A)P(A) / P(B)
    """
    # Prob_B_given_A = ((PB*PA)/PA)
    # Considering PBA equivalent to probability of B given A
    return (PBA*PA)/PB

# # ============================================================
# # Part 2 — Bernoulli Distribution
# # ============================================================

def bernoulli_pmf(x, theta):
    """
    f(x, theta) = theta^x (1-theta)^(1-x)
    """
    return ((theta**x)*((1-theta)**(1-x)))



def bernoulli_theta_analysis(theta_values):
    """
    Returns:
        (theta, P0, P1, is_symmetric)
    """
    # Considering the theta as the list of the values 
    results = []
    for theta in theta_values:
        P1 = theta
        P0 = 1 - theta
        is_symmetric = P0 == P1
        results.append([theta, P0, P1, is_symmetric])
    return results


# # ============================================================
# # Part 3 — Normal Distribution
# # ============================================================

def normal_pdf(x, mu, sigma):
    """
    Normal PDF:
        1/(sqrt(2π)σ) * exp(-(x-μ)^2 / (2σ^2))
    """
    return (1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def normal_histogram_analysis(mu_values,
                              sigma_values,
                              n_samples=10000,
                              bins=30):
    """
    For each (mu, sigma):

    Return:
        (
            mu,
            sigma,
            sample_mean,
            theoretical_mean,
            mean_error,
            sample_variance,
            theoretical_variance,
            variance_error
        )
    """
    if len(mu_values) != len(sigma_values):
        raise ValueError("mu_values and sigma_values must have the same length")

    results = []

    for mu, sigma in zip(mu_values, sigma_values):
        # Generate samples from N(mu, sigma^2)
        samples = np.random.normal(loc=mu, scale=sigma, size=n_samples)

        # Sample statistics
        sample_mean = np.mean(samples)
        sample_variance = np.var(samples)  # divide by n; for unbiased divide by n-1: np.var(samples, ddof=1)

        # Theoretical statistics
        theoretical_mean = mu
        theoretical_variance = sigma ** 2

        # Errors
        mean_error = sample_mean - theoretical_mean
        variance_error = sample_variance - theoretical_variance

        # Store results as a tuple
        results.append((
            mu,
            sigma,
            sample_mean,
            theoretical_mean,
            mean_error,
            sample_variance,
            theoretical_variance,
            variance_error
        ))

    return results
    


# # ============================================================
# # Part 4 — Uniform Distribution
# # ============================================================

def uniform_mean(a, b):
    """
    (a + b) / 2
    """
    return (a+b)/2


def uniform_variance(a, b):
    """
    (b - a)^2 / 12
    """
    return ((b-a)**2)/12


import numpy as np

def uniform_histogram_analysis(a_values, b_values, n_samples=10000, bins=30):
    """
    For each (a, b) pair:

    Returns a list of tuples:
        (
            a,
            b,
            sample_mean,
            theoretical_mean,
            mean_error,
            sample_variance,
            theoretical_variance,
            variance_error
        )
    """
    if len(a_values) != len(b_values):
        raise ValueError("a_values and b_values must have the same length")

    results = []

    for a, b in zip(a_values, b_values):
        # Generate uniform samples
        samples = np.random.uniform(low=a, high=b, size=n_samples)

        # Sample statistics
        sample_mean = np.mean(samples)
        sample_variance = np.var(samples)  # divide by n; use ddof=1 for unbiased estimate

        # Theoretical statistics
        theoretical_mean = (a + b) / 2
        theoretical_variance = ((b - a) ** 2) / 12

        # Errors
        mean_error = sample_mean - theoretical_mean
        variance_error = sample_variance - theoretical_variance

        # Store results as a tuple
        results.append((
            a,
            b,
            sample_mean,
            theoretical_mean,
            mean_error,
            sample_variance,
            theoretical_variance,
            variance_error
        ))

    return results



if __name__ == "__main__":
    print("Implement all required functions.")

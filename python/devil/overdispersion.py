"""Overdispersion parameter estimation."""

import numpy as np
from scipy import optimize
from scipy.special import digamma, polygamma
from typing import Optional, Tuple


def estimate_initial_dispersion(
    count_matrix: np.ndarray,
    offset_vector: np.ndarray
) -> np.ndarray:
    """
    Estimate initial dispersion using method of moments.
    
    Args:
        count_matrix: Count data (genes × samples).
        offset_vector: Offset values per sample.
        
    Returns:
        Initial dispersion estimates per gene.
    """
    n_genes, n_samples = count_matrix.shape
    
    # Calculate normalized mean and variance
    norm_factors = np.exp(offset_vector)
    mean_counts = np.mean(count_matrix / norm_factors, axis=1)
    var_counts = np.var(count_matrix / norm_factors, axis=1, ddof=1)
    
    # Method of moments estimator
    # Var = mu + mu^2 * dispersion
    dispersion = (var_counts - mean_counts) / (mean_counts ** 2)
    
    # Handle edge cases
    dispersion = np.where(np.isfinite(dispersion) & (dispersion > 0), 
                         dispersion, 100.0)
    
    return dispersion


def fit_dispersion(
    beta: np.ndarray,
    design_matrix: np.ndarray,
    y: np.ndarray,
    offset_vector: np.ndarray,
    tolerance: float = 1e-3,
    max_iter: int = 100,
    do_cox_reid_adjustment: bool = True
) -> float:
    """
    Fit dispersion parameter using maximum likelihood.
    
    Uses Newton-Raphson optimization with analytical gradients.
    Implementation adapted from DESeq2/glmGamPoi.
    
    Args:
        beta: Fitted coefficients for gene.
        design_matrix: Design matrix.
        y: Count vector for gene.
        offset_vector: Offset values.
        tolerance: Convergence tolerance.
        max_iter: Maximum iterations.
        do_cox_reid_adjustment: Whether to apply Cox-Reid adjustment.
        
    Returns:
        Estimated dispersion parameter.
    """
    # Handle all-zero genes
    if np.all(y == 0):
        return 0.0
    
    # Calculate mean values
    mu = np.exp(design_matrix @ beta + offset_vector)
    mu = np.maximum(mu, 1e-10)
    
    # Initial dispersion estimate
    sample_var = np.var(y)
    sample_mean = np.mean(y)
    init_disp = max((sample_var - sample_mean) / sample_mean**2, 0.1)
    
    # Define log-likelihood and derivatives
    def neg_log_likelihood(log_theta):
        """Negative log-likelihood as function of log(dispersion)."""
        theta = np.exp(log_theta)
        return -compute_nb_log_likelihood(y, mu, theta, design_matrix, 
                                        do_cox_reid_adjustment)
    
    def score(log_theta):
        """Score function (first derivative)."""
        theta = np.exp(log_theta)
        return -compute_nb_score(y, mu, theta, design_matrix, 
                               do_cox_reid_adjustment)
    
    def hessian(log_theta):
        """Hessian (second derivative)."""
        theta = np.exp(log_theta)
        return -compute_nb_hessian(y, mu, theta, design_matrix,
                                 do_cox_reid_adjustment)
    
    # Optimize using Newton-Raphson via scipy
    try:
        result = optimize.minimize(
            neg_log_likelihood,
            x0=np.log(init_disp),
            method='Newton-CG',
            jac=score,
            hess=hessian,
            options={'maxiter': max_iter, 'xtol': tolerance}
        )
    except Exception:
        return init_disp
    
    # If optimization failed, try without Cox-Reid adjustment
    if not result.success and do_cox_reid_adjustment:
        try:
            result = optimize.minimize(
                lambda x: -compute_nb_log_likelihood(y, mu, np.exp(x),
                                                   design_matrix, False),
                x0=np.log(init_disp),
                method='L-BFGS-B',
                bounds=[(np.log(1e-8), np.log(1e8))],
                options={'maxiter': max_iter, 'ftol': tolerance}
            )
        except Exception:
            return init_disp
    
    return np.exp(result.x[0]) if result.success else init_disp


def compute_nb_log_likelihood(
    y: np.ndarray,
    mu: np.ndarray,
    theta: float,
    design_matrix: np.ndarray,
    do_cox_reid: bool
) -> float:
    """
    Compute negative binomial log-likelihood.
    
    Args:
        y: Observed counts.
        mu: Expected values.
        theta: Dispersion parameter.
        design_matrix: Design matrix for Cox-Reid adjustment.
        do_cox_reid: Whether to apply Cox-Reid adjustment.
        
    Returns:
        Log-likelihood value.
    """
    from scipy.special import gammaln
    
    # Basic NB log-likelihood
    alpha = 1.0 / theta
    ll = np.sum(
        gammaln(y + alpha) - gammaln(alpha) - gammaln(y + 1) +
        alpha * np.log(alpha / (alpha + mu)) +
        y * np.log(mu / (alpha + mu))
    )
    
    # Cox-Reid adjustment
    if do_cox_reid:
        W = mu / (1 + mu * theta)
        XWX = design_matrix.T @ (W[:, np.newaxis] * design_matrix)
        sign, logdet = np.linalg.slogdet(XWX)
        ll -= 0.5 * logdet
    
    return ll


def compute_nb_score(
    y: np.ndarray,
    mu: np.ndarray,
    theta: float,
    design_matrix: np.ndarray,
    do_cox_reid: bool
) -> float:
    """Compute score function (derivative of log-likelihood)."""
    eps = 1e-6
    ll1 = compute_nb_log_likelihood(y, mu, theta - eps, design_matrix, do_cox_reid)
    ll2 = compute_nb_log_likelihood(y, mu, theta + eps, design_matrix, do_cox_reid)
    return (ll2 - ll1) / (2 * eps)


def compute_nb_hessian(
    y: np.ndarray,
    mu: np.ndarray, 
    theta: float,
    design_matrix: np.ndarray,
    do_cox_reid: bool
) -> float:
    """Compute Hessian (second derivative of log-likelihood)."""
    eps = 1e-6
    ll1 = compute_nb_log_likelihood(y, mu, theta - eps, design_matrix, do_cox_reid)
    ll2 = compute_nb_log_likelihood(y, mu, theta + eps, design_matrix, do_cox_reid)
    ll0 = compute_nb_log_likelihood(y, mu, theta, design_matrix, do_cox_reid)
    return (ll1 - 2 * ll0 + ll2) / (eps ** 2)

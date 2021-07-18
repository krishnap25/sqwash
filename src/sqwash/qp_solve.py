"""Utility to solve quadratic program in the definition of the L2-smoothed superquantile.

.. moduleauthor:: Krishna Pillutla
"""
import torch

@torch.no_grad()
def solve_qp_for_l2_smooth_superquantile(
        s: torch.Tensor, 
        superquantile_tail_fraction: float, 
        smoothing_coefficient:float
):
    """Solve quadratic program (QP) involved in the L2 smoothing of the superquantile.

    The quadratic program is 
    
    .. math:: \min \{\langle q, s\rangle - \frac{\nu}{2n}\|q - 1/n\|^2_2 \, :\, 0 \le q \le 1/(\theta n), \sum_{i=1}^n q_i = 1 \},

    where :math:`n` is the dimension of :math:`s` (and also of :math:`q`) and 
    :math:`\theta` is `superquantile_tail_fraction` and :\math:`\nu` is the smoothing coefficient.

    This QP has a special form and can be solved with a specialized efficient algorithm,
    without the need for generic QP solvers. The time and space complexity of this algorithm
    is :math:`O(n^2)`. It is designed to run efficiently on a GPU due to parallelism. 

    For details on how the QP is solved algorithmically, please see:
    Yassine Laguel and Krishna Pillutla and Jérôme Malick and Zaid Harchaoui,
    "Superquantiles at Work: Machine Learning Applications and Efficient Subgradient Computation",
    Submitted, 2021.

    The gap between the smoothed superquantile and the original superquantile is :math:`\frac{\nu}{n}`,
    which is lower order when compared to the gap :math:`O(1/\sqrt{n})` gap between the 
    empirical superquantile and the population superquantiles, as given by the law of large numbers.
    Hence, we divide the smoothing parameter by :math:`n` so that the same value of the smoothing parameter
    can be used across different batch sizes. 
    """
    zero = torch.tensor(0, dtype=s.dtype, device=s.device)
    one = torch.tensor(1, dtype=s.dtype, device=s.device)
    n = s.shape[0]
    smoothing_coefficient = smoothing_coefficient / n  # scale down
    nu_over_n = smoothing_coefficient / n
    f = (1 - superquantile_tail_fraction) / superquantile_tail_fraction
    # Solve the problem via the dual solution (the scalar :math:`\eta`)
    # Step 1: Narrow the search for eta by interval tightening
    candidate_points = torch.sort(torch.cat([s + nu_over_n, s - nu_over_n * f]))[0] # (2*n,)
    # phiprime is the derivative of the dual 
    phiprime_at_candidate_points = 1 - torch.where(
            s[None, :] >= candidate_points[:, None] + nu_over_n * f, 
            one / (n * superquantile_tail_fraction),
            torch.where(
                s[None, :] <= candidate_points[:, None] - nu_over_n, 
                zero, 
                (s[None, :] - candidate_points[:, None]) / smoothing_coefficient + 1/n
            )
        ).sum(axis=1)  # (2*n,)
    # Find the largest point with negative phiprime and the smallest point with positive phiprime
    lower_idx = candidate_points[phiprime_at_candidate_points <= 0].argmax()
    upper_idx = candidate_points[phiprime_at_candidate_points >= 0].argmin()
    eta_lower = candidate_points[phiprime_at_candidate_points <= 0][lower_idx] # \underline \eta
    eta_upper = candidate_points[phiprime_at_candidate_points >= 0][upper_idx] # \overline \eta
    # Compute phiprime at eta_lower and eta_upper
    phiprime_lower = phiprime_at_candidate_points[phiprime_at_candidate_points <= 0][lower_idx]
    phiprime_upper = phiprime_at_candidate_points[phiprime_at_candidate_points >= 0][upper_idx]
    # Step 2: Find the dual solution (eta_star) by linearly interpolating between eta_lower and eta_upper
    if phiprime_lower == 0:
        eta_star = eta_lower
    elif phiprime_upper == 0:
        eta_star = eta_upper
    else:
        eta_star = (
            eta_lower - 
            phiprime_lower * (eta_upper - eta_lower) / (phiprime_upper - phiprime_lower)
        )
    # Step 3: Construct primal solution from dual solution in closed form
    primal_solution = torch.where(
            s >= eta_star + nu_over_n * f, 
            one / (n * superquantile_tail_fraction),
            torch.where(
                s <= eta_star - nu_over_n, 
                zero, 
            (s - eta_star) / smoothing_coefficient + 1/n
            )
        )
    # measure primal infeasibility (which is also the dual derivative) for debugging
    # it should be <1e-6 at least
    # primal_infeasibility = torch.abs(primal_solution.sum() - 1) 
    return primal_solution

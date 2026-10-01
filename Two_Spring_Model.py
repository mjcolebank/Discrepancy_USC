"""
coupled_model.py

Standalone module containing:
 - coupled_sys(yall, t, params, m, Fterms): ODE right-hand side for the coupled system
 - call_model(params, t_space, F_param): helper that integrates the ODE and returns
   the solution as a stacked torch tensor (matching your original return shape).

Notes:
 - Uses scipy.integrate.odeint for integration.
 - Accepts params either as a numpy array-like or a PyTorch tensor (detaches internally).
 - Preserves original comments and major methods from the user-provided code.
"""

from typing import Sequence, Tuple, Union
import numpy as np
from scipy.integrate import odeint
import torch


def coupled_sys(yall: Union[np.ndarray, Sequence],
                t: float,
                params: Union[np.ndarray, Sequence],
                m: Sequence[float],
                Fterms: Sequence[float]):
    """
    RHS for the coupled second-order system expanded with sensitivities.

    Arguments:
    - yall: full state vector (length 20):
        (x1,x2,v1,v2,
         x1k1,x2k1,v1k1,v2k1,
         x1c1,x2c1,v1c1,v2c1,
         x1k2,x2k2,v1k2,v2k2,
         x1c2,x2c2,v1c2,v2c2)
    - t: time (scalar)
    - params: parameter vector [k1, c1, k2, c2] (array-like)
    - m: mass vector [m1, m2]
    - Fterms: forcing terms tuple/list (F0, omega_f) — currently F2 set to zero in code

    Returns:
    - dYall: numpy array length 20, time-derivative of yall
    """
    # unpack state (8 states + 16 sensitivities = 24? but original code uses 20 total)
    # Original mapping (as provided):
    (x1, x2, v1, v2,
     x1k1, x2k1, v1k1, v2k1,
     x1c1, x2c1, v1c1, v2c1,
     x1k2, x2k2, v1k2, v2k2,
     x1c2, x2c2, v1c2, v2c2) = yall

    # Sensitivities are packed after the first 4 states (original code)
    sens = np.asarray(yall[4:])  # length 16

    F0, omega_f = Fterms

    # Unpack parameters
    k1, c1, k2, c2 = params
    # k1, c1, k2, c2 = np.exp(params)
    m1, m2 = m

    # Jacobian of system (per-4-block repeated 4 times => 16x16)
    dgdy = np.zeros((16, 16), dtype=float)
    for i in range(4):
        ind = 4 * i
        # position derivatives map to velocities
        dgdy[ind, ind + 2] = 1.0
        dgdy[ind + 1, ind + 3] = 1.0

        # acceleration block entries (for the i-th 4-state block)
        dgdy[ind + 2, ind] = -(k1 + k2) / m1
        dgdy[ind + 2, ind + 1] = k2 / m1
        dgdy[ind + 2, ind + 2] = -(c1 + c2) / m1
        dgdy[ind + 2, ind + 3] = c2 / m1

        dgdy[ind + 3, ind] = k2 / m2
        dgdy[ind + 3, ind + 1] = -k2 / m2
        dgdy[ind + 3, ind + 2] = c2 / m2
        dgdy[ind + 3, ind + 3] = -c2 / m2

    # Sensitivity-to-parameter matrix dgdpar (1 row per RHS eq? original code used a single 1x4 array)
    # The original code builds a single row of 4 columns wrapped as a 1x4 array, then later adds it to ds.
    # Here we preserve the original pattern: a 1x4 array (but later broadcasting will apply).
    dgdpar = np.array([[0, 0, -x1 / m1, 0,
                        0, 0, -v1 / m1, 0,
                        0, 0, (-x1 + x2) / m1, (x1 - x2) / m2,
                        0, 0, (-v1 + v2) / m1, (v1 - v2) / m2]],
                      dtype=float)

    # RHS equations
    F2 = F0 * np.cos(omega_f * t)  #(kept zero consistent with original)
    # velocity derivatives
    dx1 = v1
    dx2 = v2

    # acceleration equations (explicit scalar form)
    dv1 = (-(c1 + c2) * v1 + c2 * v2 - (k1 + k2) * x1 + k2 * x2) / m1
    dv2 = (c2 * v1 - c2 * v2 + k2 * x1 - k2 * x2 + F2) / m2

    # Sensitivity ODEs: ds = dgdy * sens + dgdpar
    # sens is a vector of length 16; dgdy is 16x16; dgdpar in original code is (1,16),
    # so we will interpret dgdpar as a 1x16 row and add it to the result (broadcast).
    ds = dgdy.dot(sens) + dgdpar.ravel()

    # Assemble derivative vector of full state
    dYall = np.zeros(20, dtype=float)
    dYall[0:4] = [dx1, dx2, dv1, dv2]
    dYall[4:] = ds  # length 16

    return dYall


def call_model(params: Union[np.ndarray, torch.Tensor],
               t_space: Union[np.ndarray, Sequence],
               F_param: Tuple[float, float]):
    """
    Integrate the coupled system over the time grid t_space.

    Arguments:
    - params: array-like of parameters (k1, c1, k2, c2) or a torch tensor (will be detached)
    - t_space: 1D array of time points to integrate over
    - F_param: tuple (F0, omega_f) passed as Fterms to coupled_sys

    Returns:
    - torch.stack of 20 tensors (each tensor has length len(t_space)), matching original code
    """
    # initial conditions (20 states)
    X0 = np.zeros(20, dtype=float)
    X0[0] = 5.0

    # masses (hard-coded as in original)
    masses = [5.0, 10.0]

    # Convert params to numpy array if it's a torch tensor
    if isinstance(params, torch.Tensor):
        params_np = params.detach().cpu().numpy()
    else:
        params_np = np.asarray(params, dtype=float)

    tspace = np.asarray(t_space, dtype=float)

    # Solve the system with scipy.integrate.odeint
    solution = odeint(coupled_sys, X0, tspace, args=(params_np, masses, F_param))

    # Unpack solution columns (each is a vector of length len(tspace))
    (x1, x2, v1, v2,
     x1k1, x2k1, v1k1, v2k1,
     x1c1, x2c1, v1c1, v2c1,
     x1k2, x2k2, v1k2, v2k2,
     x1c2, x2c2, v1c2, v2c2) = solution.T

    # Convert numpy arrays to torch tensors and stack (preserve time-series)
    # Use torch.from_numpy for efficiency and dtype preservation
    tensors = [
        torch.from_numpy(x1), torch.from_numpy(x2),
        torch.from_numpy(v1), torch.from_numpy(v2),
        torch.from_numpy(x1k1), torch.from_numpy(x2k1),
        torch.from_numpy(v1k1), torch.from_numpy(v2k1),
        torch.from_numpy(x1c1), torch.from_numpy(x2c1),
        torch.from_numpy(v1c1), torch.from_numpy(v2c1),
        torch.from_numpy(x1k2), torch.from_numpy(x2k2),
        torch.from_numpy(v1k2), torch.from_numpy(v2k2),
        torch.from_numpy(x1c2), torch.from_numpy(x2c2),
        torch.from_numpy(v1c2), torch.from_numpy(v2c2)
    ]

    return torch.stack(tensors)


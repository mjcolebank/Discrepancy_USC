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
    (x1,x2,x3,x4,v1,v2,v3,v4,
     x1k1,x2k1,x3k1,x4k1,v1k1,v2k1,v3k1,v4k1,
     x1c1,x2c1,x3c1,x4c1,v1c1,v2c1,v3c1,v4c1,
     x1k2,x2k2,x3k2,x4k2,v1k2,v2k2,v3k2,v4k2,
     x1c2,x2c2,x3c2,x4c2,v1c2,v2c2,v3c2,v4c2,
     x1k3,x2k3,x3k3,x4k3,v1k3,v2k3,v3k3,v4k3,
     x1c3,x2c3,x3c3,x4c3,v1c3,v2c3,v3c3,v4c3,
     x1k4,x2k4,x3k4,x4k4,v1k4,v2k4,v3k4,v4k4,
     x1c4,x2c4,x3c4,x4c4,v1c4,v2c4,v3c4,v4c4) = yall

    # Sensitivities are packed after the first 4 states (original code)
    sens = np.asarray(yall[8:])  # length 16

    F0, omega_f = Fterms

    # Unpack parameters
    # k1, c1, k2, c2 = params
    k1,c1,k2,c2,k3,c3,k4,c4 = params
    # k1,c1,k2,c2,k3,c3,k4,c4 = np.exp(params)
    m1,m2,m3,m4 = m

    # Jacobian of system
    dgdy = np.zeros((64,64))
    for i in range(8):
        ind = 8*i
        dgdy[ind,ind+4]     = 1.0
        dgdy[ind+1,ind+5]   = 1.0
        dgdy[ind+2,ind+6]   = 1.0
        dgdy[ind+3,ind+7]   = 1.0

        dgdy[ind+4,ind]   = -(k1+k2)/m1
        dgdy[ind+4,ind+1] = k2/m1
        dgdy[ind+4,ind+4] = -(c1+c2)/m1
        dgdy[ind+4,ind+5] = c2/m1
        
        dgdy[ind+5,ind]   = k2/m2
        dgdy[ind+5,ind+1] = -(k2+k3)/m2
        dgdy[ind+5,ind+2] = k3/m2
        dgdy[ind+5,ind+4] = c2/m2
        dgdy[ind+5,ind+5] = -(c2+c3)/m2
        dgdy[ind+5,ind+6] = c3/m2

        dgdy[ind+6,ind+1]   = k3/m3
        dgdy[ind+6,ind+2] = -(k3+k4)/m3
        dgdy[ind+6,ind+3] = k4/m3
        dgdy[ind+6,ind+5] = c3/m3
        dgdy[ind+6,ind+6] = -(c3+c4)/m3
        dgdy[ind+6,ind+7] = c4/m3

        dgdy[ind+7,ind+2] = k4/m4
        dgdy[ind+7,ind+3] = -k4/m4
        dgdy[ind+7,ind+6] = c4/m4
        dgdy[ind+7,ind+7] = -c4/m4 

        
    # Sensitivity vector should be column vector defined by dRhs1/dpar1, dRHS2/dpar1, dRHS3/dpar1, .... up to dRHS_N / dpar_N
    dgdpar = np.array([[0,0,0,0,-x1/m1,0,0,0,
                        0,0,0,0,-v1/m1,0,0,0,
                        0,0,0,0,(-x1+x2)/m1,(x1-x2)/m2,0,0,
                        0,0,0,0,(-v1+v2)/m1,(v1-v2)/m2,0,0,
                        0,0,0,0,0,(-x2+x3)/m2,(x2-x3)/m3,0,
                        0,0,0,0,0,(-v2+v3)/m2,(v2-v3)/m3,0,
                        0,0,0,0,0,0,(-x3+x4)/m3,(x3-x4)/m4,
                        0,0,0,0,0,0,(-v3+v4)/m3,(v3-v4)/m4]],
                        dtype=float)

    # RHS equations
    F = F0 * np.cos(omega_f * t)  #(kept zero consistent with original)
    # velocity derivatives (trivial)
    dx1 = v1
    dx2 = v2
    dx3 = v3
    dx4 = v4


    # acceleration equations (explicit scalar form)
    dv1 = ( - (c1 + c2)*v1 +   c2*v2  - (k1 + k2)*x1 +   k2*x2 ) / m1
    dv2 = (   c2*v1 - (c2 + c3)*v2 + c3*v3 + k2*x1 - (k2 + k3)*x2 + k3*x3) / m2
    dv3 = (   c3*v2 - (c3 + c4)*v3 + c4*v4  + k3*x2 - (k3 + k4)*x3 + k4*x4) / m3
    dv4 = (   c4*v3 - c4*v4 + k4*x3 - k4*x4 + F ) / m4

    ds = np.matmul(dgdy,sens) + dgdpar

    dYall = np.zeros(72,dtype=float)
    dYall[0:8] = [dx1,dx2,dx3,dx4,dv1,dv2,dv3,dv4]
    dYall[8:] = ds
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
    X0 = np.zeros(72, dtype=float)
    X0[0] = 5.0

    # masses (hard-coded as in original)
    masses = [5.0,4.0,7.0,10.0]

    # Convert params to numpy array if it's a torch tensor
    if isinstance(params, torch.Tensor):
        params_np = params.detach().cpu().numpy()
    else:
        params_np = np.asarray(params, dtype=float)

    tspace = np.asarray(t_space, dtype=float)

    # Solve the system with scipy.integrate.odeint
    solution = odeint(coupled_sys, X0, tspace, args=(params_np, masses, F_param))

    # Unpack solution columns (each is a vector of length len(tspace))
    (x1,x2,x3,x4,v1,v2,v3,v4,
     x1k1,x2k1,x3k1,x4k1,v1k1,v2k1,v3k1,v4k1,
     x1c1,x2c1,x3c1,x4c1,v1c1,v2c1,v3c1,v4c1,
     x1k2,x2k2,x3k2,x4k2,v1k2,v2k2,v3k2,v4k2,
     x1c2,x2c2,x3c2,x4c2,v1c2,v2c2,v3c2,v4c2,
     x1k3,x2k3,x3k3,x4k3,v1k3,v2k3,v3k3,v4k3,
     x1c3,x2c3,x3c3,x4c3,v1c3,v2c3,v3c3,v4c3,
     x1k4,x2k4,x3k4,x4k4,v1k4,v2k4,v3k4,v4k4,
     x1c4,x2c4,x3c4,x4c4,v1c4,v2c4,v3c4,v4c4) = solution.T

    # Convert numpy arrays to torch tensors and stack (preserve time-series)
    # Use torch.from_numpy for efficiency and dtype preservation
    tensors = [
        torch.tensor(x1), torch.tensor(x2), torch.tensor(x3), torch.tensor(x4),
            torch.tensor(v1), torch.tensor(v2), torch.tensor(v3), torch.tensor(v4),
            torch.tensor(x1k1), torch.tensor(x2k1), torch.tensor(x3k1), torch.tensor(x4k1),
            torch.tensor(v1k1), torch.tensor(v2k1), torch.tensor(v3k1), torch.tensor(v4k1),
            torch.tensor(x1c1), torch.tensor(x2c1), torch.tensor(x3c1), torch.tensor(x4c1),
            torch.tensor(v1c1), torch.tensor(v2c1), torch.tensor(v3c1), torch.tensor(v4c1),
            torch.tensor(x1k2), torch.tensor(x2k2), torch.tensor(x3k2), torch.tensor(x4k2),
            torch.tensor(v1k2), torch.tensor(v2k2), torch.tensor(v3k2), torch.tensor(v4k2),
            torch.tensor(x1c2), torch.tensor(x2c2), torch.tensor(x3c2), torch.tensor(x4c2),
            torch.tensor(v1c2), torch.tensor(v2c2), torch.tensor(v3c2), torch.tensor(v4c2),
            torch.tensor(x1k3), torch.tensor(x2k3), torch.tensor(x3k3), torch.tensor(x4k3),
            torch.tensor(v1k3), torch.tensor(v2k3), torch.tensor(v3k3), torch.tensor(v4k3),
            torch.tensor(x1c3), torch.tensor(x2c3), torch.tensor(x3c3), torch.tensor(x4c3),
            torch.tensor(v1c3), torch.tensor(v2c3), torch.tensor(v3c3), torch.tensor(v4c3),
            torch.tensor(x1k4), torch.tensor(x2k4), torch.tensor(x3k4), torch.tensor(x4k4),
            torch.tensor(v1k4), torch.tensor(v2k4), torch.tensor(v3k4), torch.tensor(v4k4),
            torch.tensor(x1c4), torch.tensor(x2c4), torch.tensor(x3c4), torch.tensor(x4c4),
            torch.tensor(v1c4), torch.tensor(v2c4), torch.tensor(v3c4), torch.tensor(v4c4) 
    ]

    return torch.stack(tensors)


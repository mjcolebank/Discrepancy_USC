

import torch 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm, multivariate_normal, invgamma, uniform

def coupled_sys(t,yall,params):
    v, w, va, wa, vb, wb, vtau, wtau, vI, wI = yall
    a, b, tau, I = params

    inv_tau = 1.0 / tau

    # Original system
    dvdt = v - v**3 / 3.0 - w + I
    dwdt = (v + a - b * w) * inv_tau

    # Common Jacobian entries
    J11 = 1.0 - v * v
    J12 = -1.0
    J21 = inv_tau
    J22 = -b * inv_tau

    # Sensitivity wrt a
    dvadt = J11 * va + J12 * wa
    dwadt = J21 * va + J22 * wa + inv_tau

    # Sensitivity wrt b
    dvbdt = J11 * vb + J12 * wb
    dwbdt = J21 * vb + J22 * wb - w * inv_tau

    # Sensitivity wrt tau
    dvtau_dt = J11 * vtau + J12 * wtau
    dwtau_dt = J21 * vtau + J22 * wtau - dwdt * inv_tau

    # Sensitivity wrt I
    dvI_dt = J11 * vI + J12 * wI + 1.0
    dwI_dt = J21 * vI + J22 * wI

    return np.array([
        dvdt, dwdt,
        dvadt, dwadt,
        dvbdt, dwbdt,
        dvtau_dt, dwtau_dt,
        dvI_dt, dwI_dt
    ])


def call_model(params, x):
    """
    params : torch.Tensor or numpy array of shape (4,)  (log-params in your code)
    x      : torch.Tensor or numpy array with time points (len = n_obs)
    Returns:
      a torch.Tensor of shape (24, n_obs) on the SAME device as `params` if params is a tensor,
      otherwise returns on CPU.
    """
    # decide device for returned tensor
    if torch.is_tensor(params):
        out_device = params.device
        params_np = params.detach().cpu().numpy()
    else:
        out_device = torch.device('cpu')
        params_np = np.asarray(params)

    if torch.is_tensor(x):
        tspace = x.detach().cpu().numpy()
    else:
        tspace = np.asarray(x)

    # initial conditions and N as in your original function
    v0 = -1.0
    w0 = 1.0
    X0 = np.zeros(10)
    X0[0:2] = [v0,w0]  

    # call SciPy's odeint (works on CPU numpy arrays)
    solution = odeint(coupled_sys, X0, tspace, args=(params_np,))  # shape (n_time, 24)

    # transpose -> (10, n_time), convert to torch, move to out_device
    sol_t = torch.from_numpy(solution.y.astype(np.float32)).to(device=out_device) 
    
    return sol_t   # shape (10, n_obs) on device out_device

# def ode_jac(t,);
     

def compute_jacobian(theta,x):
        # Break Jacobian into numerical solution of system 
        v,w,va,wa,vb,wb,vtau,wtau,vI,wI  = call_model(theta,x)
        # f1,f2,s11,s21,s12,s22 = solution.T
        # print(np.size(Smu,axis=0))
        jac = torch.empty(2, va.shape[0], 4, device=va.device, dtype=va.dtype)        
        jac[0,:,0] = va
        jac[0,:,1] = vb
        jac[0,:,2] = vtau
        jac[0,:,3] = vI

        jac[1,:,0] = wa
        jac[1,:,1] = wb
        jac[1,:,2] = wtau
        jac[1,:,3] = wI


        
        return jac    


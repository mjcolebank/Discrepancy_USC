

import torch 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm, multivariate_normal, invgamma, uniform


def coupled_sys(yall,t,params,N):
    # 5 states and 5x4=20 sensitivities
    S,I,R,V,Sa,Ia,Ra,Va,Sv,Iv,Rv,Vv,Smu,Imu,Rmu,Vmu,Swr,Iwr,Rwr,Vwr,Swv,Iwv,Rwv,Vwv = yall
    sens = yall[4:]
    # Unpack parameters
    # a,v,mu,wr,wv = np.exp(params)
    a,v,mu,wr,wv = params
    # Jacobian of system
    dgdy = np.zeros((20,20))
    for i in range(5):
        ind = 4*i
        dgdy[ind,ind]   = -a*I/N - v
        dgdy[ind,ind+1] = -a*S/N 
        dgdy[ind,ind+2] = wr
        dgdy[ind,ind+3] = wv 
        

        dgdy[ind+1,ind]   = a*I/N
        dgdy[ind+1,ind+1] = a*S/N - mu

        dgdy[ind+2,ind+1] = mu
        dgdy[ind+2,ind+2] = -wr
        
        dgdy[ind+3,ind] = v
        dgdy[ind+3,ind+3] = -wv
        

        
    # Sensitivity vector should be column vector defined by dRhs1/dpar1, dRHS2/dpar1, dRHS3/dpar1, .... up to dRHS_N / dpar_N
    dgdpar = np.array([[-S*I/N,S*I/N,0,0,-S,0,0,S,0,-I,I,0,R,0,-R,0,V,0,0,-V]])

    # RHS equations
    dSdt = -a*S*I/N - v*S + wr*R + wv*V
    dIdt = a*S*I/N - mu*I 
    dRdt = mu*I - wr*R
    dVdt = v*S - wv*V

    ds = np.matmul(dgdy,sens) + dgdpar

    dYall = np.zeros(24)
    dYall[0:4] = [dSdt,dIdt,dRdt,dVdt]
    dYall[4:] = ds
    return dYall


def call_model(params, x):
    """
    params : torch.Tensor or numpy array of shape (5,)  (log-params in your code)
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
    S0 = 99999
    I0 = 1
    R0 = 0
    V0 = 0
    N = S0 + I0 + R0 + V0
    X0 = np.zeros(24)
    X0[0:4] = [S0, I0, R0, V0]

    # call SciPy's odeint (works on CPU numpy arrays)
    solution = odeint(coupled_sys, X0, tspace, args=(params_np, N))  # shape (n_time, 24)

    # transpose -> (24, n_time), convert to torch, move to out_device
    sol_t = torch.from_numpy(solution.T.astype(np.float32)).to(device=out_device) / float(N)

    return sol_t   # shape (24, n_obs) on device out_device

# def call_model(params,x):
#     S0 = 99999
#     I0 = 1
#     R0 = 0
#     V0 = 0
#     N = S0+I0+R0+V0
#     X0 = np.zeros(24)
#     X0[0:4] = S0,I0,R0,V0

#     # Define the total population based on the initial conditions



#     # Parameters as a list

#     # Define the end time for the numerical solution and the time points
#     tspace = x
#     # Solve the system
#     print(params)
#     solution = odeint(coupled_sys, X0, tspace, args=(params.detach().cpu().numpy(), N))


#     # Unpack solution and return model + sensitivities scaled by the total population
#     S,I,R,V,Sa,Ia,Ra,Va,Sv,Iv,Rv,Vv,Smu,Imu,Rmu,Vmu,Swr,Iwr,Rwr,Vwr,Swv,Iwv,Rwv,Vwv = solution.T
#     return torch.stack([torch.tensor(S),torch.tensor(I),torch.tensor(R),torch.tensor(V),
#                         torch.tensor(Sa),torch.tensor(Ia),torch.tensor(Ra),torch.tensor(Va),
#                         torch.tensor(Sv),torch.tensor(Iv),torch.tensor(Rv),torch.tensor(Vv),
#                         torch.tensor(Smu),torch.tensor(Imu),torch.tensor(Rmu),torch.tensor(Vmu),
#                         torch.tensor(Swr),torch.tensor(Iwr),torch.tensor(Rwr),torch.tensor(Vwr),
#                         torch.tensor(Swv),torch.tensor(Iwv),torch.tensor(Rwv),torch.tensor(Vwv)])/N


def compute_jacobian(theta,x):
        # Break Jacobian into numerical solution of system 
        S,I,R,V,Sa,Ia,Ra,Va,Sv,Iv,Rv,Vv,Smu,Imu,Rmu,Vmu,Swr,Iwr,Rwr,Vwr,Swv,Iwv,Rwv,Vwv  = call_model(theta,x)
        # f1,f2,s11,s21,s12,s22 = solution.T
        # print(np.size(Smu,axis=0))
        jac = torch.empty(4, Smu.shape[0], 5, device=Smu.device, dtype=Smu.dtype)        
        jac[:,0,0] = Sa
        jac[:,0,1] = Sv
        jac[:,0,2] = Smu
        jac[:,0,3] = Swr
        jac[:,0,4] = Swv

        jac[:,1,0] = Ia
        jac[:,1,1] = Iv
        jac[:,1,2] = Imu
        jac[:,1,3] = Iwr
        jac[:,1,4] = Iwv

        jac[:,2,0] = Ra
        jac[:,2,1] = Rv
        jac[:,2,2] = Rmu
        jac[:,2,3] = Rwr
        jac[:,2,4] = Rwv

        jac[:,3,0] = Va
        jac[:,3,1] = Vv
        jac[:,3,2] = Vmu
        jac[:,3,3] = Vwr
        jac[:,3,4] = Vwv

        
        return jac    


# def compute_jacobian_IRonly(theta,x):
#         # Break Jacobian into numerical solution of system 
#         S,I,R,V,Sa,Ia,Ra,Va,Sv,Iv,Rv,Vv,Smu,Imu,Rmu,Vmu,Swr,Iwr,Rwr,Vwr,Swv,Iwv,Rwv,Vwv  = call_model(theta,x)
#         # f1,f2,s11,s21,s12,s22 = solution.T
#         # print(np.size(Smu,axis=0))
#         jac = torch.empty(3, Smu.shape[0], 5, device=Smu.device, dtype=Smu.dtype)        
#         jac[0,:,0] = Ia
#         jac[1,:,0] = Iv
#         jac[2,:,0] = Imu
#         jac[3,:,0] = Iwr
#         jac[4,:,0] = Iwv

#         jac[0,:,1] = Ra
#         jac[1,:,1] = Rv
#         jac[2,:,1] = Rmu
#         jac[3,:,1] = Rwr
#         jac[4,:,1] = Rwv

    
        
#         return jac  

def compute_jacobian_IRVonly(theta,x):
        # Break Jacobian into numerical solution of system 
        S,I,R,V,Sa,Ia,Ra,Va,Sv,Iv,Rv,Vv,Smu,Imu,Rmu,Vmu,Swr,Iwr,Rwr,Vwr,Swv,Iwv,Rwv,Vwv  = call_model(theta,x)
        # f1,f2,s11,s21,s12,s22 = solution.T
        # print(np.size(Smu,axis=0))
        # jac = torch.empty(np.size(Smu,axis=0),3,5)
        
        # jac[:,0,0] = Ia
        # jac[:,0,1] = Iv
        # jac[:,0,2] = Imu
        # jac[:,0,3] = Iwr
        # jac[:,0,4] = Iwv

        # jac[:,1,0] = Ra
        # jac[:,1,1] = Rv
        # jac[:,1,2] = Rmu
        # jac[:,1,3] = Rwr
        # jac[:,1,4] = Rwv

        # jac[:,2,0] = Va
        # jac[:,2,1] = Vv
        # jac[:,2,2] = Vmu
        # jac[:,2,3] = Vwr
        # jac[:,2,4] = Vwv


        jac = torch.empty(3,np.size(Smu,axis=0),5)
        
        jac[0,:,0] = Ia
        jac[0,:,1] = Iv
        jac[0,:,2] = Imu
        jac[0,:,3] = Iwr
        jac[0,:,4] = Iwv

        jac[1,:,0] = Ra
        jac[1,:,1] = Rv
        jac[1,:,2] = Rmu
        jac[1,:,3] = Rwr
        jac[1,:,4] = Rwv

        jac[2,:,0] = Va
        jac[2,:,1] = Vv
        jac[2,:,2] = Vmu
        jac[2,:,3] = Vwr
        jac[2,:,4] = Vwv

    
        
        return jac  

def compute_jacobian_IVonly(theta,x):
        # Break Jacobian into numerical solution of system 
        S,I,R,V,Sa,Ia,Ra,Va,Sv,Iv,Rv,Vv,Smu,Imu,Rmu,Vmu,Swr,Iwr,Rwr,Vwr,Swv,Iwv,Rwv,Vwv  = call_model(theta,x)
        # f1,f2,s11,s21,s12,s22 = solution.T
        # print(np.size(Smu,axis=0))
        # jac = torch.empty(np.size(Smu,axis=0),3,5)
        
        # jac[:,0,0] = Ia
        # jac[:,0,1] = Iv
        # jac[:,0,2] = Imu
        # jac[:,0,3] = Iwr
        # jac[:,0,4] = Iwv

        # jac[:,1,0] = Ra
        # jac[:,1,1] = Rv
        # jac[:,1,2] = Rmu
        # jac[:,1,3] = Rwr
        # jac[:,1,4] = Rwv

        # jac[:,2,0] = Va
        # jac[:,2,1] = Vv
        # jac[:,2,2] = Vmu
        # jac[:,2,3] = Vwr
        # jac[:,2,4] = Vwv


        jac = torch.empty(2,np.size(Smu,axis=0),5)
        
        jac[0,:,0] = Ia
        jac[0,:,1] = Iv
        jac[0,:,2] = Imu
        jac[0,:,3] = Iwr
        jac[0,:,4] = Iwv

        jac[1,:,0] = Va
        jac[1,:,1] = Vv
        jac[1,:,2] = Vmu
        jac[1,:,3] = Vwr
        jac[1,:,4] = Vwv

    
        
        return jac  
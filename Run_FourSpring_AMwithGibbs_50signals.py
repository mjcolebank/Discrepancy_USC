"""
Concise four-spring calibration script based on FourSpring_AMwithGibbs.ipynb.

Changes from the notebook:
  1. Runs as a single Python script.
  2. Generates 50 random true parameter vectors by default.
  3. For each true signal, generates observations, runs multi-start optimization,
     then runs adaptive Metropolis-with-Gibbs MCMC for IND and MOP discrepancy models.
  4. Includes a local fallback adaptive_metropolis_s2update implementation if
     AM_withGibbs.py is not available.

Required local files:
  - Four_Spring_Model.py

Example:
  python FourSpring_AMwithGibbs_50signals.py --outdir results_50signals

Quick smoke test:
  python FourSpring_AMwithGibbs_50signals.py --n-signals 1 --n-grid 40 --n-obs 8 --mcmc-steps 20 --burn-adapt 5 --n-starts 2
"""

import argparse
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from scipy import interpolate
from scipy.optimize import least_squares
from scipy.stats import invgamma, qmc

from Four_Spring_Model import call_model

try:
    from AM_withGibbs import adaptive_metropolis_s2update as external_amwg
except Exception:  # noqa: BLE001 - optional user module
    external_amwg = None


J = 4
P = 8
LOG2PI = float(np.log(2.0 * np.pi))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def as_float_tensor(x) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def compute_jitter(cmat: torch.Tensor, target_min_eigval=1e-6, min_jitter=1e-10) -> float:
    eigvals = torch.linalg.eigvalsh(0.5 * (cmat + cmat.T))
    lambda_min = eigvals.min()
    return float(torch.clamp(torch.tensor(target_min_eigval, dtype=cmat.dtype) - lambda_min, min=min_jitter))


def safe_cholesky(cov: torch.Tensor, min_jitter=1e-8, max_tries=8):
    cov = 0.5 * (cov + cov.T)
    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    jitter = min_jitter
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(cov + jitter * eye), jitter
        except RuntimeError:
            jitter *= 10.0
    # Final attempt with eigenvalue-based jitter.
    jitter = compute_jitter(cov, target_min_eigval=min_jitter, min_jitter=jitter)
    return torch.linalg.cholesky(cov + jitter * eye), jitter


def model_states(theta, t, force) -> torch.Tensor:
    """Return first four displacement states in task-major order: shape (J, n_time)."""
    y_all = call_model(as_float_tensor(theta), as_float_tensor(t), np.asarray(force, dtype=float))
    return y_all[:J, :].to(torch.float32)


def model_flat(theta, t, force) -> torch.Tensor:
    return model_states(theta, t, force).reshape(-1)


def compute_jacobian(theta, t, force) -> torch.Tensor:
    """
    Sensitivity tensor matching the notebook layout.
    Returns shape (J, n_time, P), where P = [k1,c1,k2,c2,k3,c3,k4,c4].
    """
    y_all = call_model(as_float_tensor(theta), as_float_tensor(t), np.asarray(force, dtype=float))
    y_all = y_all.to(torch.float32)
    n = y_all.shape[1]
    jac = torch.empty((J, n, P), dtype=torch.float32)
    for p in range(P):
        # call_model packs each parameter sensitivity as
        # [x1p,x2p,x3p,x4p,v1p,v2p,v3p,v4p].
        jac[:, :, p] = y_all[8 + 8 * p : 8 + 8 * p + J, :]
    return jac


def rbf_cov(t: torch.Tensor, lengthscale: torch.Tensor) -> torch.Tensor:
    t = as_float_tensor(t).reshape(-1, 1)
    lengthscale = torch.clamp(as_float_tensor(lengthscale), min=1e-6)
    d2 = torch.cdist(t, t) ** 2
    return torch.exp(-0.5 * d2 / (lengthscale**2))


def multitask_rbf_cov(t: torch.Tensor, gamma: torch.Tensor, lengthscale: torch.Tensor) -> torch.Tensor:
    """Independent-task RBF covariance in task-major flattening order."""
    k_t = rbf_cov(t, lengthscale)
    return as_float_tensor(gamma) * torch.kron(torch.eye(J, dtype=torch.float32), k_t)


def orthogonal_cov(theta, t, force, gamma, lengthscale, min_jitter=1e-8) -> torch.Tensor:
    """MOP covariance C - C J (J' C J)^-1 J' C, using notebook projection logic."""
    t = as_float_tensor(t)
    n = t.numel()
    m = n * J
    xvol = 1.0 / float(n)

    c = multitask_rbf_cov(t, gamma, lengthscale)
    jac = compute_jacobian(theta, t, force)          # (J, n, P)
    jmat = jac.reshape(m, P)                         # task-major, same as notebook
    cj = c @ jmat
    hmat = (xvol * xvol) * (jmat.T @ cj)
    chol_h, _ = safe_cholesky(hmat, min_jitter=min_jitter)
    h = xvol * cj
    proj = h @ torch.cholesky_solve(h.T, chol_h)
    c_delta = c - proj
    return 0.5 * (c_delta + c_delta.T)


def mvn_log_prob_from_cholesky(y, mean, cov, min_jitter=1e-8) -> torch.Tensor:
    y = as_float_tensor(y).reshape(-1)
    mean = as_float_tensor(mean).reshape(-1)
    cov = as_float_tensor(cov)
    r = (y - mean).reshape(-1, 1)
    chol, _ = safe_cholesky(cov, min_jitter=min_jitter)
    alpha = torch.cholesky_solve(r, chol)
    quad = (r.T @ alpha).squeeze()
    logdet = 2.0 * torch.sum(torch.log(torch.diagonal(chol)))
    return -0.5 * (quad + logdet + y.numel() * LOG2PI)


def sample_mop_discrepancy(theta_true, tvals, force, gamma_true, bw_true) -> torch.Tensor:
    cov = orthogonal_cov(theta_true, tvals, force, gamma_true, bw_true, min_jitter=1e-8)
    chol, _ = safe_cholesky(cov, min_jitter=1e-8)
    z = torch.randn(tvals.numel() * J, dtype=torch.float32)
    return (chol @ z).view(J, tvals.numel()).T.detach()  # shape (n_grid, J)


def make_observations(theta_true, tvals, t_obs, force_true, force_approx, gamma_true, bw_true, sigma_e):
    delta = sample_mop_discrepancy(theta_true, tvals, force_approx, gamma_true, bw_true)
    f_true = model_states(theta_true, tvals, force_true).T  # (n_grid, J)
    xi_true = f_true + delta

    obs_cols = []
    t_np = tvals.detach().cpu().numpy()
    tobs_np = t_obs.detach().cpu().numpy()
    for j in range(J):
        interp_j = interpolate.interp1d(t_np, xi_true[:, j].detach().cpu().numpy())
        obs_cols.append(torch.from_numpy(interp_j(tobs_np)).float().view(-1, 1))
    xi_obs = torch.cat(obs_cols, dim=1)
    y_obs = xi_obs + torch.normal(0.0, sigma_e, size=xi_obs.shape)
    return y_obs.to(torch.float32), xi_true.to(torch.float32), xi_obs.to(torch.float32), delta.to(torch.float32)


def sobol_starts(lb, ub, n_starts, seed=0):
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    if n_starts > 0 and (n_starts & (n_starts - 1) == 0):
        sampler = qmc.Sobol(d=len(lb), scramble=True, seed=seed)
        sample = sampler.random_base2(m=int(np.log2(n_starts)))
    else:
        rng = np.random.default_rng(seed)
        sample = rng.random((n_starts, len(lb)))
    return qmc.scale(sample, lb, ub)


def optimize_signal(y_obs, t_obs, force, lb, ub, n_starts=50, seed=0, max_nfev=500):
    y_flat = y_obs.T.reshape(-1).detach().cpu().numpy().astype(float)
    starts = sobol_starts(lb, ub, n_starts, seed=seed)
    results = []

    def residual(theta_np):
        pred = model_flat(theta_np, t_obs, force).detach().cpu().numpy().astype(float)
        return y_flat - pred

    for i, theta0 in enumerate(starts):
        res = least_squares(
            residual,
            x0=np.asarray(theta0, dtype=float),
            bounds=(np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)),
            method="trf",
            diff_step=1e-3,
            max_nfev=max_nfev,
        )
        results.append(
            {
                "run": i,
                "x": res.x,
                "fun": float(np.sum(res.fun**2)),
                "cost": float(res.cost),
                "success": bool(res.success),
                "message": str(res.message),
                "nfev": int(res.nfev),
            }
        )
    order = np.argsort([r["fun"] for r in results])
    x_sorted = np.vstack([results[i]["x"] for i in order])
    f_sorted = np.array([results[i]["fun"] for i in order])
    return results, x_sorted, f_sorted, starts


def make_potential_functions(y_obs, t_obs, force_approx):
    y_flat = y_obs.T.reshape(-1).to(torch.float32)
    m = y_flat.numel()
    eye_m = torch.eye(m, dtype=torch.float32)

    def potential_fn_ind(theta, gptheta, _gp_cov, s2):
        gamma, bw = as_float_tensor(gptheta)
        mean = model_flat(theta, t_obs, force_approx)
        cov = multitask_rbf_cov(t_obs, gamma, bw) + as_float_tensor(s2) * eye_m
        logp = mvn_log_prob_from_cholesky(y_flat, mean, cov)
        ss = torch.sum((y_flat - mean) ** 2)
        return logp, ss

    def potential_fn_mop(theta, gptheta, _gp_cov, s2):
        gamma, bw = as_float_tensor(gptheta)
        mean = model_flat(theta, t_obs, force_approx)
        c_delta = orthogonal_cov(theta, t_obs, force_approx, gamma, bw)
        cov = c_delta + as_float_tensor(s2) * eye_m
        logp = mvn_log_prob_from_cholesky(y_flat, mean, cov)
        ss = torch.sum((y_flat - mean) ** 2)
        return logp, ss

    return potential_fn_ind, potential_fn_mop


def local_adaptive_metropolis_s2update(
    post_func,
    theta0,
    k0,
    M,
    covar,
    UB,
    LB,
    GP_cov=None,
    s2_init=0.01,
    n_obs=None,
    a0=1e-3,
    b0=1e-3,
):
    """Fallback AM-with-Gibbs sampler compatible with the notebook call signature."""
    theta0 = as_float_tensor(theta0)
    UB = as_float_tensor(UB)
    LB = as_float_tensor(LB)
    covar = as_float_tensor(covar)
    d = theta0.numel()
    chain = torch.empty((d, M), dtype=torch.float32)
    lik_chain = torch.empty(M, dtype=torch.float32)
    s2_chain = torch.empty(M, dtype=torch.float32)

    theta = torch.clamp(theta0.clone(), LB, UB)
    s2 = float(s2_init)
    chol_prop, _ = safe_cholesky(covar + 1e-12 * torch.eye(d), min_jitter=1e-12)

    def eval_post(th, s2_val):
        return post_func(th[:P], th[P:], GP_cov, torch.tensor(s2_val, dtype=torch.float32))

    logp, ss = eval_post(theta, s2)
    samples_for_adapt = []

    for m in range(M):
        if m > k0 and len(samples_for_adapt) > d + 5:
            hist = torch.stack(samples_for_adapt)
            emp_cov = torch.cov(hist.T)
            scaled = (2.38**2 / d) * emp_cov + 1e-8 * torch.eye(d)
            chol_prop, _ = safe_cholesky(scaled, min_jitter=1e-10)

        prop = theta + chol_prop @ torch.randn(d)
        in_bounds = bool(torch.all(prop >= LB) and torch.all(prop <= UB))
        if in_bounds:
            prop_logp, prop_ss = eval_post(prop, s2)
            if torch.isfinite(prop_logp):
                log_alpha = prop_logp - logp
                if float(torch.log(torch.rand(1))) < float(log_alpha):
                    theta, logp, ss = prop, prop_logp, prop_ss

        # Gibbs update for measurement variance from residual sum of squares.
        n_data = int(n_obs) * J if n_obs is not None else int(ss.numel())
        shape = a0 + 0.5 * n_data
        scale = b0 + 0.5 * float(ss)
        s2 = float(invgamma.rvs(a=shape, scale=scale))

        chain[:, m] = theta
        lik_chain[m] = logp
        s2_chain[m] = s2
        samples_for_adapt.append(theta.clone())

    return chain, lik_chain, s2_chain


def run_mcmc(y_obs, t_obs, force_approx, theta_true, x_best, gamma_true, bw_true, lb, ub, sigma_e, args):
    potential_ind, potential_mop = make_potential_functions(y_obs, t_obs, force_approx)

    ub_mcmc = torch.cat((as_float_tensor(ub), torch.tensor([gamma_true*1.5, bw_true*1.5], dtype=torch.float32)))#torch.tensor([ub.flatten(), 1.5 * gamma_true, 1.5 * bw_true])
    lb_mcmc = torch.cat((as_float_tensor(lb), torch.tensor([gamma_true*0.5, bw_true*0.5], dtype=torch.float32)))#torch.tensor([lb.flatten(), 0.5 * gamma_true, 0.5 * bw_true])
    theta_all_true = torch.cat((as_float_tensor(theta_true), torch.tensor([gamma_true, bw_true], dtype=torch.float32)))
    proposal_sd = 0.01 * (ub_mcmc - lb_mcmc)
    covar = torch.diag(proposal_sd**2)

    data_flat = y_obs.T.reshape(-1)
    output = model_flat(x_best, t_obs, force_approx)
    s20_init = torch.sum((data_flat - output) ** 2) / data_flat.numel()
    if not torch.isfinite(s20_init) or s20_init <= 0:
        s20_init = torch.tensor(sigma_e**2, dtype=torch.float32)

    sampler = external_amwg if (external_amwg is not None and not args.force_local_mcmc) else local_adaptive_metropolis_s2update
    chain_ind_all = torch.empty((args.n_chains, len(theta_all_true), args.mcmc_steps), dtype=torch.float32)
    chain_mop_all = torch.empty_like(chain_ind_all)
    s2_ind_all = torch.empty((args.n_chains, args.mcmc_steps), dtype=torch.float32)
    s2_mop_all = torch.empty_like(s2_ind_all)

    for c in range(args.n_chains):
        par_init = theta_all_true + proposal_sd * torch.randn_like(theta_all_true)
        par_init = torch.max(torch.min(par_init, ub_mcmc), lb_mcmc)
        chain_ind, _, s2_ind = sampler(
            post_func=potential_ind,
            theta0=par_init,
            k0=args.burn_adapt,
            M=args.mcmc_steps,
            covar=covar,
            UB=ub_mcmc,
            LB=lb_mcmc,
            GP_cov=None,
            s2_init=s20_init,
            n_obs=y_obs.shape[0],
        )
        chain_mop, _, s2_mop = sampler(
            post_func=potential_mop,
            theta0=par_init,
            k0=args.burn_adapt,
            M=args.mcmc_steps,
            covar=covar,
            UB=ub_mcmc,
            LB=lb_mcmc,
            GP_cov=None,
            s2_init=s20_init,
            n_obs=y_obs.shape[0],
        )
        chain_ind_all[c] = chain_ind
        chain_mop_all[c] = chain_mop
        s2_ind_all[c] = s2_ind
        s2_mop_all[c] = s2_mop

    return {
        "theta_all_true": theta_all_true,
        "UB_MCMC": ub_mcmc,
        "LB_MCMC": lb_mcmc,
        "proposal_sd": proposal_sd,
        "s20_init": s20_init,
        "chain_IND_all": chain_ind_all,
        "chain_MOP_all": chain_mop_all,
        "s2chain_IND_all": s2_ind_all,
        "s2chain_MOP_all": s2_mop_all,
    }


def tensor_to_numpy_dict(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        return {k: tensor_to_numpy_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tensor_to_numpy_dict(v) for v in obj]
    return obj


def json_safe(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="Four-spring 50-signal optimization + AM-with-Gibbs MCMC")
    parser.add_argument("--n-signals", type=int, default=50)
    parser.add_argument("--seed", type=int, default=325)
    parser.add_argument("--n-grid", type=int, default=200)
    parser.add_argument("--n-obs", type=int, default=20)
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--tf", type=float, default=10.0)
    parser.add_argument("--sigma-e", type=float, default=0.1)
    parser.add_argument("--F0", type=float, default=25.0)
    parser.add_argument("--omega-f", type=float, default=5.0)
    parser.add_argument("--n-starts", type=int, default=2)
    parser.add_argument("--max-nfev", type=int, default=500)
    parser.add_argument("--mcmc-steps", type=int, default=10000)
    parser.add_argument("--burn-adapt", type=int, default=2000)
    parser.add_argument("--n-chains", type=int, default=1)
    parser.add_argument("--outdir", type=str, default="FourSpring_50signal_results")
    parser.add_argument("--force-local-mcmc", action="store_true", help="Ignore AM_withGibbs.py even if available.")
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tvals = torch.linspace(args.t0, args.tf, args.n_grid, dtype=torch.float32)
    t_obs = torch.linspace(args.t0, args.tf, args.n_obs, dtype=torch.float32)
    force_true = np.array([args.F0, args.omega_f], dtype=float)
    force_approx = force_true.copy()

    lb = np.array([1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1], dtype=float)
    ub = np.array([100.0, 10.0, 100.0, 10.0, 100.0, 10.0, 100.0, 10.0], dtype=float)
    rng = np.random.default_rng(args.seed)
    theta_true_values = rng.uniform(lb, ub, size=(args.n_signals, P))

    manifest = {
        "args": vars(args),
        "theta_true_values": theta_true_values.tolist(),
        "files": [],
    }

    for s, theta_true in enumerate(theta_true_values):
        # Log normal generation of Discrepancy parameters
        gamma_true = rng.lognormal(mean=-2.0,sigma=1.0)
        bw_true    = rng.lognormal(mean=-0.2,sigma=0.8)
        signal_seed = args.seed + 1000 * s
        set_seed(signal_seed)
        print(f"\n=== Signal {s + 1}/{args.n_signals} | seed={signal_seed} ===")
        print("theta_true =", np.array2string(theta_true, precision=4))

        y_obs, xi_true, xi_obs, delta_true = make_observations(
            theta_true=theta_true,
            tvals=tvals,
            t_obs=t_obs,
            force_true=force_true,
            force_approx=force_approx,
            gamma_true=gamma_true,
            bw_true=bw_true,
            sigma_e=args.sigma_e,
        )

        opt_results, x_sorted, f_sorted, start_thetas = optimize_signal(
            y_obs=y_obs,
            t_obs=t_obs,
            force=force_approx,
            lb=lb,
            ub=ub,
            n_starts=args.n_starts,
            seed=signal_seed,
            max_nfev=args.max_nfev,
        )
        print("best objective =", float(f_sorted[0]))
        print("best theta     =", np.array2string(x_sorted[0], precision=4))

        mcmc_results = run_mcmc(
            y_obs=y_obs,
            t_obs=t_obs,
            force_approx=force_approx,
            theta_true=theta_true,
            x_best=x_sorted[0],
            gamma_true=gamma_true,
            bw_true=bw_true,
            lb = lb,
            ub = ub,
            sigma_e=args.sigma_e,
            args=args,
        )

        result = {
            "signal_index": s,
            "seed": signal_seed,
            "tvals": tvals,
            "t_obs": t_obs,
            "theta_true": as_float_tensor(theta_true),
            "force_true": force_true,
            "force_approx": force_approx,
            "gamma_true": gamma_true,
            "bw_true": bw_true,
            "sigma_e": args.sigma_e,
            "obs": y_obs,
            "xi_true": xi_true,
            "xi_obs": xi_obs,
            "delta_true": delta_true,
            "start_thetas": start_thetas,
            "optimization_results": opt_results,
            "x_sorted": x_sorted,
            "f_sorted": f_sorted,
            **mcmc_results,
        }

        pkl_path = outdir / f"signal_{s:03d}_results.pkl"
        with pkl_path.open("wb") as f:
            pickle.dump(tensor_to_numpy_dict(result), f)

        npz_path = outdir / f"signal_{s:03d}_results.npz"
        np.savez_compressed(npz_path, **tensor_to_numpy_dict(result))

        manifest["files"].append({"signal_index": s, "pkl": str(pkl_path), "npz": str(npz_path)})
        with (outdir / "manifest.json").open("w") as f:
            json.dump(json_safe(manifest), f, indent=2)

    print(f"\nDone. Wrote results and manifest to: {outdir}")


if __name__ == "__main__":
    main()

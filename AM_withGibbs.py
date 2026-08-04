import torch

def adaptive_metropolis_s2update(
    post_func,
    theta0,
    M,
    covar,
    UB,
    LB,
    GP_cov,
    k0=100,
    s2_init=1e-3,
    n_obs=1,
):
    """
    Adaptive Metropolis for raw-scale parameters:

        z = [a, v, mu, wr, wv, gamma, bw, s2]

    post_func should return:

        log_like, ss = post_func(thetamodel, thetaGP, GP_cov, s2)

    where:

        thetamodel = z[:5]
        thetaGP    = z[5:7]
        s2         = z[7]
    """
    alpha_s2=2.0
    beta_s2=1e-3

    def sample_s2_from_inverse_gamma(ss):

        alpha_post = torch.as_tensor(
            alpha_s2 + 0.5 * n_obs,
            dtype=theta0.dtype,
            device=theta0.device,
        )

        beta_post = torch.as_tensor(
            beta_s2,
            dtype=theta0.dtype,
            device=theta0.device,
        ) + 0.5 * ss

        gamma_sample = torch.distributions.Gamma(
            concentration=alpha_post,
            rate=beta_post,
        ).sample()

        return 1.0 / gamma_sample

    n_par = len(theta0)

    chain = torch.zeros((n_par, M), dtype=theta0.dtype, device=theta0.device)
    s2_chain = torch.zeros(M, dtype=theta0.dtype, device=theta0.device)

    logp_chain = torch.zeros(M, dtype=theta0.dtype, device=theta0.device)

    if covar is None or len(covar) == 0:
        scale = 0.1 * torch.abs(theta0) + 1e-6
        covar = torch.diag(scale ** 2).to(dtype=theta0.dtype, device=theta0.device)
    else:
        covar = covar.to(dtype=theta0.dtype, device=theta0.device)

    UB = UB.to(dtype=theta0.dtype, device=theta0.device)
    LB = LB.to(dtype=theta0.dtype, device=theta0.device)

    def logpost(z,s2):
        if torch.any(z > UB) or torch.any(z < LB):
            # print('poor bounds')
            return torch.tensor(-torch.inf, dtype=z.dtype, device=z.device), torch.inf
        thetamodel = z[:n_par-2]
        thetaGP = z[n_par-2:n_par]

        if torch.any(thetamodel <= 0.0) or torch.any(thetaGP <= 0.0) or s2 <= 0.0:
            # print('bad pars')
            return torch.tensor(-torch.inf, dtype=z.dtype, device=z.device), torch.inf


        try:
            log_like, ss = post_func(thetamodel, thetaGP, GP_cov, s2)

            if not torch.isfinite(log_like):
                # print('not finite')
                return torch.tensor(-torch.inf, dtype=z.dtype, device=z.device), torch.inf

            # Add priors here if post_func only returns likelihood.
            # For example, weak uniform priors are already implied by bounds.
            return log_like, ss

        except Exception:
            # print('hit exception')
            return torch.tensor(-torch.inf, dtype=z.dtype, device=z.device), torch.inf

    chain[:, 0] = theta0
    s2 = torch.as_tensor(s2_init, dtype=theta0.dtype, device=theta0.device)
    s2_chain[0] = s2
    logp_old, ss_old = logpost(theta0,s2=s2)
    logp_chain[0] = logp_old

    eps = 1e-8
    sp_cov = (2.38 ** 2) / n_par

    eye = torch.eye(n_par, dtype=theta0.dtype, device=theta0.device)
    Vk = covar.clone()
    proposal_covar = sp_cov * Vk + eps * eye
    R = torch.linalg.cholesky(proposal_covar)

    num_acc = 0

    for i in range(1, M):
        z = torch.randn(n_par, dtype=theta0.dtype, device=theta0.device)

        # If R is lower-triangular Cholesky of covariance,
        # either R @ z or z @ R.T is fine. For vector form, use R @ z.
        theta_star = chain[:, i - 1] + R @ z

        logp_star, ss_star = logpost(theta_star,s2=s2)

        log_alpha = logp_star - logp_old

        if torch.log(torch.rand((), dtype=theta0.dtype, device=theta0.device)) < log_alpha:
            chain[:, i] = theta_star
            logp_old = logp_star
            ss_old = ss_star
            num_acc += 1
        else:
            chain[:, i] = chain[:, i - 1]
        
        # Gibbs update for s2 using the accepted/current theta.
        s2 = sample_s2_from_inverse_gamma(ss_old)
        s2_chain[i] = s2

        # Recompute log posterior at the same theta but new s2.
        # This keeps logp_old consistent for the next MH step.
        logp_old, ss_old = logpost(chain[:, i], s2)
        logp_chain[i] = logp_old

        if i % 1000 == 0:
            print("Percent done:", i / M * 100)
            print("Acceptance rate:", 100 * num_acc / i)
            print("Current log posterior:", logp_old)

        if i % k0 == 0 and i > n_par:
            X = chain[:, : i + 1].T
            Vk = torch.cov(X.T)

            proposal_covar = sp_cov * Vk + eps * eye

            try:
                R = torch.linalg.cholesky(proposal_covar)
            except RuntimeError:
                # Keep old R if adaptation produces a bad covariance.
                pass

    print("Final acceptance rate:")
    print(100 * num_acc / (M - 1))

    return chain, logp_chain, s2_chain
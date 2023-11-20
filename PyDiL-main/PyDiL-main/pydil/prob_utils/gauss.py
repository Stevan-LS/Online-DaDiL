import torch
import math


def gauss_density_1d(x, mean=0, std=1, log=True):
    log_prob = -0.5 * (((x - mean) / std) ** 2 +
                       math.log(2 * math.pi) +
                       torch.log(std))
    if log:
        return log_prob
    return torch.exp(log_prob)


def gmm_density_1d(x, means=None, stds=None, weights=None, log=True):
    if means is None:
        means = torch.Tensor([0., 0.])
    if stds is None:
        stds = torch.Tensor([1., 1.])
    if weights is None:
        weights = torch.Tensor([1 / 2, 1 / 2])

    log_probs = -0.5 * (((x[:, None] - means[None, :]) / stds[None, :]) ** 2 +
                        math.log(2 * math.pi) +
                        torch.log(stds)[None, :]) + torch.log(weights)
    log_probs = torch.logsumexp(log_probs, dim=1)
    if log:
        return log_probs
    return torch.exp(log_probs)


def __full_gauss_density_nd(x, mean=None, cov=None, log=True):
    if mean is None:
        mean = torch.zeros(x.shape[1])

    if cov is None:
        cov = torch.diag(torch.ones(x.shape[1]))

    log_prob = -0.5 * ((x - mean) @ torch.linalg.inv(cov) @ (x - mean).T +
                       torch.log(torch.linalg.det(cov)) +
                       x.shape[1] * math.log(2 * math.pi))

    if log:
        return log_prob
    return torch.exp(log_prob)


def __diag_gauss_density_nd(x, mean=None, cov=None, log=True):
    if mean is None:
        mean = torch.zeros(x.shape[1])

    if cov is None:
        cov = torch.ones(x.shape[1])

    log_prob = - 0.5 * ((((x - mean) ** 2) / cov).sum(dim=1) +
                        x.shape[1] * math.log(2 * math.pi) +
                        torch.log(torch.prod(cov)))

    if log:
        return log_prob
    return torch.exp(log_prob)


def __full_gmm_density(x, means=None, covs=None, weights=None, log=True):
    if weights is None:
        weights = torch.ones(2) / 2

    if means is None:
        means = torch.zeros(2, x.shape[1])

    if covs is None:
        covs = torch.stack([torch.eye(x.shape[1]) for _ in range(2)])

    inv_cov = torch.linalg.inv(covs)
    dist = torch.einsum('knd,kdD,knD->kn',
                        x.unsqueeze(0) - means.unsqueeze(1),
                        inv_cov,
                        x.unsqueeze(0) - means.unsqueeze(1))
    log_probs = (-0.5 * (dist +
                         x.shape[1] * math.log(2 * math.pi) +
                         torch.log(torch.linalg.det(covs)).unsqueeze(1)) +
                 torch.log(weights).unsqueeze(1))
    log_probs = torch.logsumexp(log_probs, dim=0)
    if log:
        return log_probs
    return torch.exp(log_probs)


def __diag_gmm_density(x, means=None, covs=None, weights=None, log=True):
    if weights is None:
        weights = torch.ones(2) / 2

    if means is None:
        means = torch.zeros(2, x.shape[1])

    if covs is None:
        covs = torch.ones(2, x.shape[1])

    dist = (
        ((x.unsqueeze(0) - means.unsqueeze(1)) / covs.unsqueeze(1))
    ).sum(dim=-1)
    log_probs = (-0.5 * (dist +
                         x.shape[1] * math.log(2 * math.pi) +
                         torch.log(torch.prod(covs, dim=1)).unsqueeze(1)) +
                 torch.log(weights).unsqueeze(1))
    log_probs = torch.logsumexp(log_probs, dim=0)
    if log:
        return log_probs
    return torch.exp(log_probs)


def gauss_density_nd(x, mean=None, cov=None, log=True, cov_type='diag'):
    if cov_type == 'full':
        return __full_gauss_density_nd(x, mean, cov, log=log)
    elif cov_type == 'diag':
        return __diag_gauss_density_nd(x, mean, cov, log=log)


def gmm_density_nd(x,
                   means=None,
                   covs=None,
                   weights=None,
                   log=True,
                   cov_type='diag'):
    if cov_type == 'full':
        return __full_gmm_density(x, means, covs, weights, log=log)
    elif cov_type == 'diag':
        return __diag_gmm_density(x, means, covs, weights, log=log)

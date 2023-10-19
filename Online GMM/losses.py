import torch
from utils import sqrtm


def full_parametric_bures_wasserstein(src_mean, tgt_mean, src_cov, tgt_cov):
    sqrt_src_cov = sqrtm(src_cov)

    M = sqrtm(torch.mm(sqrt_src_cov, torch.mm(tgt_cov, sqrt_src_cov)))
    bures_metric = torch.trace(src_cov) + torch.trace(tgt_cov) - 2 * torch.trace(M)

    return torch.sqrt(torch.dist(src_mean, tgt_mean, p=2) ** 2 + bures_metric)


def diag_parametric_bures_wasserstein(src_mean, tgt_mean, src_std, tgt_std):
    return torch.sqrt(torch.linalg.norm(src_mean - tgt_mean) + torch.linalg.norm(src_std - tgt_std))


class BuresWassersteinMetric(torch.nn.Module):
    def __init__(self, cov_type='full'):
        self.cov_type = cov_type.lower()
        super(BuresWassersteinMetric, self).__init__()

    def forward(self, xs, xt):
        μs = xs.mean(dim=0)
        μt = xt.mean(dim=0)
        if self.cov_type.lower() == 'diag':
            σs = xs.std(dim=0)
            σt = xt.std(dim=0)
            return diag_parametric_bures_wasserstein(μs, μt, σs, σt)
        else:
            Σs = torch.cov(xs.T)
            Σt = torch.cov(xt.T)
            return full_parametric_bures_wasserstein(μs, μt, Σs, Σt)

import torch


def parametric_bures_wasserstein_metric(meanP,
                                        covP,
                                        meanQ,
                                        covQ,
                                        cov_type='full',
                                        item=False):
    mean_diff = torch.linalg.norm(meanP - meanQ) ** 2

    if cov_type == 'full':
        DP, VP = torch.linalg.eig(covP)
        DP = torch.diag(DP ** (1 / 2))
        covP_sqrt = (VP @ DP @ VP.T).real

        D, V = torch.linalg.eig(covP_sqrt @ covQ @ covP_sqrt)
        D = torch.diag(D ** (1 / 2))
        M = (V @ D @ V.T).real

        cov_diff = torch.trace(covP) + torch.trace(covQ) - 2 * torch.trace(M)
    elif cov_type == 'commute':
        DP, VP = torch.linalg.eig(covP)
        DP = torch.diag(DP ** (1 / 2))
        covP_sqrt = (VP @ DP @ VP.T).real

        DQ, VQ = torch.linalg.eig(covQ)
        DQ = torch.diag(DQ ** (1 / 2))
        covQ_sqrt = (VQ @ DQ @ VQ.T).real

        cov_diff = torch.linalg.norm(covP_sqrt - covQ_sqrt, ord='fro')
    elif cov_type == 'diag':
        cov_diff = torch.linalg.norm(covP - covQ) ** 2
    if item:
        return torch.sqrt(mean_diff + cov_diff).item()
    return torch.sqrt(mean_diff + cov_diff)

import torch
import numpy as np
from scipy.stats import multivariate_normal


a = 0.8

def split(gauss):
    eigenvalues, eigenvectors = torch.linalg.eigh(gauss[3])
    lamb_i = torch.argmax(eigenvalues).item()
    nu = eigenvectors[:, lamb_i]
    delta_nu = torch.sqrt(a*eigenvalues[lamb_i]) * nu
    tau = gauss[0]/2
    c = gauss[1]/2
    mu1 = gauss[2] + delta_nu
    mu2 = gauss[2] - delta_nu
    sigma = gauss[3] - torch.outer(delta_nu, delta_nu)
    return [tau, c, mu1, sigma], [tau, c, mu2, sigma]

def merge(gauss1, gauss2):
    merged_weight = gauss1[0] + gauss2[0]
    merged_c = gauss1[1] + gauss2[1]
    f1 = gauss1[0]/merged_weight
    f2 = gauss2[0]/merged_weight
    merged_mean = f1 * gauss1[2] + f2 * gauss2[2]
    merged_cov = f1 * gauss1[3] + f2 * gauss2[3] \
                + f1 * f2 * torch.outer(gauss1[2] - gauss2[2], gauss1[2] - gauss2[2])
    merged_gauss = [merged_weight, merged_c, merged_mean, merged_cov]
    return merged_gauss

def sqrtm(A, return_inv=False):
    D, V = torch.linalg.eig(A)

    A_sqrt = torch.mm(V, torch.mm(torch.diag(D.pow(1 / 2)), V.T)).real
    if return_inv:
        A_sqrt_neg = torch.mm(V, torch.mm(torch.diag(D.pow(-1 / 2)), V.T)).real
        return A_sqrt, A_sqrt_neg
    return A_sqrt

def bures_wasserstein_metric(mean1, mean2, cov1, cov2):
    sqrt_src_cov = sqrtm(cov1)

    M = sqrtm(torch.mm(sqrt_src_cov, torch.mm(cov2, sqrt_src_cov)))
    bures_metric = torch.trace(cov1) + torch.trace(cov2) - 2 * torch.trace(M)

    return torch.sqrt(torch.dist(mean1, mean2, p=2) ** 2 + bures_metric)

def init_cov_and_T_split(K_max, range, n):
    cov_init_size = range/(2*K_max)
    cov_init = cov_init_size*torch.eye(n)
    T_split = 2*torch.det(cov_init).item()
    return cov_init, T_split

def normal_pdf(x, mean, cov):
    return (2*np.pi)**(-len(x)/2) * np.linalg.det(cov)**(-1/2) * np.exp(-1/2 * (x - mean).T @ np.linalg.inv(cov) @ (x - mean))

def OGMM(X, K_max, sigma0, alpha, T_split):
    GMM = []
    for i in range(len(X)):
        new_x = X[i]
        P = torch.zeros(len(GMM))
        for j in range(len(GMM)):
            P[j] = GMM[j][0] * normal_pdf(new_x, mean=GMM[j][2], cov=GMM[j][3])
        if len(GMM) < K_max:
            GMM.append([alpha, 1, new_x, sigma0])
        else:
            Q = P/torch.sum(P).item()
            for j in range(len(GMM)):
                GMM[j][1] = GMM[j][1] + Q[j]
                GMM[j][0] = (1 - alpha) * GMM[j][0] + alpha * Q[j]
                eta_j = Q[j]*((1-alpha)/GMM[j][1] + alpha)
                GMM[j][2], GMM[j][3] = (1 - eta_j)*GMM[j][2] + eta_j*new_x, \
                                        (1 - eta_j)*GMM[j][3] + eta_j*torch.outer(new_x - GMM[j][2], new_x - GMM[j][2])
        total_weight = np.sum([GMM[j][0] for j in range(len(GMM))])
        for j in range(len(GMM)):
            GMM[j][0] = GMM[j][0]/total_weight
        
        # Split and merge
        if len(GMM) > 0:
            
            # Split
            V = [torch.det(GMM[j][3]) for j in range(len(GMM))]
            s = np.argmax(V)
            if V[s] > T_split:
                gauss_split1, gauss_split2 = split(GMM[s])
                del GMM[s]
                GMM.append(gauss_split1)
                GMM.append(gauss_split2)
            
            # Merge            
            while len(GMM) > K_max:
                bwm = torch.full((len(GMM), len(GMM)), torch.inf)
                for i in range(len(GMM)):
                    for j in range(i):
                        gauss1 = GMM[i]
                        gauss2 = GMM[j]
                        mean1 = gauss1[2]
                        mean2 = gauss2[2]
                        cov1 = gauss1[3]
                        cov2 = gauss2[3]
                        bwm[i, j] = bures_wasserstein_metric(mean1, mean2, cov1, cov2)
                min_value = torch.min(bwm).item()
                min_indexes = torch.argwhere(bwm == min_value)[0]
                merged_gauss = merge(GMM[min_indexes[0]], GMM[min_indexes[1]])
                del GMM[max(min_indexes[0], min_indexes[1])]
                del GMM[min(min_indexes[0], min_indexes[1])]
                GMM.append(merged_gauss)   
        

    return GMM
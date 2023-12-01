import torch
import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import IncrementalPCA


class Online_GMM(torch.nn.Module):

    def __init__(self, n_components, lr, n_features, data_range, batch_size):
            self.lr = lr
            self.n_components = n_components
            self.n_features = n_features
            self.data_range = data_range
            self.cov_init, self.T_split = self.init_cov_and_T_split(self.n_components, self.data_range, self.n_features)
            self.a = 0.8
            self.weights = []
            self.c = []
            self.means = []
            self.cov = []
            self.batch_size = batch_size
            self.ipca = None
    
    def split_gaussian(self, gauss_number):
        eigenvalues, eigenvectors = torch.linalg.eigh(self.cov[gauss_number])
        lamb_i = torch.argmax(eigenvalues).item()
        nu = eigenvectors[:, lamb_i]
        delta_nu = torch.sqrt(self.a*eigenvalues[lamb_i]) * nu
        tau = self.weights[gauss_number]/2
        c = self.c[gauss_number]/2
        mu1 = self.means[gauss_number] + delta_nu
        mu2 = self.means[gauss_number] - delta_nu
        sigma = self.cov[gauss_number] - torch.outer(delta_nu, delta_nu)
        del self.weights[gauss_number]
        del self.c[gauss_number]
        del self.means[gauss_number]
        del self.cov[gauss_number]
        self.weights.extend([tau, tau])
        self.c.extend([c, c])
        self.means.extend([mu1, mu2])
        self.cov.extend([sigma, sigma])

    def merge_gaussian(self, gauss_number1, gauss_number2):
        merged_weight = self.weights[gauss_number1] + self.weights[gauss_number2]
        merged_c = self.c[gauss_number1] + self.c[gauss_number2]
        f1 = self.weights[gauss_number1]/merged_weight
        f2 = self.weights[gauss_number2]/merged_weight
        merged_mean = f1 * self.means[gauss_number1] + f2 * self.means[gauss_number2]
        merged_cov = f1 * self.cov[gauss_number1] + f2 * self.cov[gauss_number2] \
                    + f1 * f2 * torch.outer(self.means[gauss_number1] - self.means[gauss_number2], self.means[gauss_number1] - self.means[gauss_number2])
        max_gauss_number = max(gauss_number1, gauss_number2)
        min_gauss_number = min(gauss_number1, gauss_number2)
        del self.weights[max_gauss_number]
        del self.weights[min_gauss_number]
        del self.c[max_gauss_number]
        del self.c[min_gauss_number]
        del self.means[max_gauss_number]
        del self.means[min_gauss_number]
        del self.cov[max_gauss_number]
        del self.cov[min_gauss_number]
        self.weights.append(merged_weight)
        self.c.append(merged_c)
        self.means.append(merged_mean)
        self.cov.append(merged_cov)

    def sqrtm(self, A, return_inv=False):
        D, V = torch.linalg.eig(A)

        A_sqrt = torch.mm(V, torch.mm(torch.diag(D.pow(1 / 2)), V.T)).real
        if return_inv:
            A_sqrt_neg = torch.mm(V, torch.mm(torch.diag(D.pow(-1 / 2)), V.T)).real
            return A_sqrt, A_sqrt_neg
        return A_sqrt

    def bures_wasserstein_metric(self, mean1, mean2, cov1, cov2):
        sqrt_src_cov = self.sqrtm(cov1)

        M = self.sqrtm(torch.mm(sqrt_src_cov, torch.mm(cov2, sqrt_src_cov)))
        bures_metric = torch.trace(cov1) + torch.trace(cov2) - 2 * torch.trace(M)

        return torch.sqrt(torch.dist(mean1, mean2, p=2) ** 2 + bures_metric)

    def init_cov_and_T_split(self, n_components, data_range, n):
        cov_init_size = data_range/(2*n_components)
        cov_init = cov_init_size*torch.eye(n)
        T_split = 2*torch.det(cov_init).item()
        return cov_init, T_split

    def normal_pdf(self, x, mean, cov):
        return (2*np.pi)**(-len(x)/2) * np.linalg.det(cov)**(-1/2) * np.exp(-1/2 * (x - mean).T @ np.linalg.inv(cov) @ (x - mean))

    def fit_sample(self, X, dimension_reduction=False):
        if dimension_reduction:
            if self.ipca == None:
                self.ipca = IncrementalPCA(n_components=self.n_features)
            self.ipca.partial_fit(X)
            X = torch.from_numpy(self.ipca.transform(X)).float()

        for i in range(X.shape[0]):
            new_x = X[i]
            n_current_components = len(self.weights)
            P = torch.zeros(n_current_components)
            for j in range(n_current_components):
                P[j] = self.weights[j] * self.normal_pdf(new_x, mean=self.means[j], cov=self.cov[j])
            
            '''if n_current_components > 0 and torch.sum(P) == 0:
                dist = [torch.dist(new_x, self.means[k], p=2) for k in range(n_current_components)]
                P[np.argmin(dist)] = 1'''
            
            if n_current_components < self.n_components:
                self.weights.append(self.lr)
                self.c.append(1)
                self.means.append(new_x)
                self.cov.append(self.cov_init)
            else:
                Q = P/torch.sum(P).item()
                for j in range(n_current_components):
                    self.c[j] = self.c[j] + Q[j]
                    self.weights[j] = (1 - self.lr) * self.weights[j] + self.lr * Q[j]
                    eta_j = Q[j]*((1-self.lr)/self.c[j] + self.lr)
                    self.means[j], self.cov[j] = (1 - eta_j)*self.means[j] + eta_j*new_x, \
                                            (1 - eta_j)*self.cov[j] + eta_j*torch.outer(new_x - self.means[j], new_x - self.means[j])
                    
            total_weight = np.sum([self.weights[j] for j in range(len(self.weights))])
            for j in range(len(self.weights)):
                self.weights[j] = self.weights[j]/total_weight
            
            # Split and merge
            if len(self.weights) > 0:
                
                # Split
                V = [torch.det(self.cov[j]) for j in range(len(self.cov))]
                s = np.argmax(V)
                if V[s] > self.T_split:
                    self.split_gaussian(s)
                
                # Merge            
                while len(self.weights) > self.n_components:
                    bwm = torch.full((len(self.weights), len(self.weights)), torch.inf)
                    for i in range(len(self.weights)):
                        for j in range(i):
                            mean1 = self.means[i]
                            mean2 = self.means[j]
                            cov1 = self.cov[i]
                            cov2 = self.cov[j]
                            bwm[i, j] = self.bures_wasserstein_metric(mean1, mean2, cov1, cov2)
                    min_value = torch.min(bwm).item()
                    min_indexes = torch.argwhere(bwm == min_value)[0]
                    self.merge_gaussian(min_indexes[0], min_indexes[1])

    def sample(self, n_samples=None):
        n_samples = self.batch_size if n_samples is None else n_samples
        if np.sum(np.array(self.weights)[:-1].astype(np.float64)) > 1:
            self.weights = [self.weights[i]/1.000001 for i in range(len(self.weights))]
        n_samples_comp = np.random.multinomial(n_samples, np.array(self.weights))
        X = np.vstack(
                [
                    np.random.multivariate_normal(mean, covariance, int(sample))
                    for (mean, covariance, sample) in zip(
                        self.means, self.cov, n_samples_comp
                    )
                ]
            )
        if self.ipca is not None:
            X = self.ipca.inverse_transform(X) 
        return torch.from_numpy(X).float(), None

    def generate_list(self):
        return [[self.weights[i], self.c[i],self.means[i], self.cov[i]] for i in range(len(self.weights))]
    
    def log_normal_pdf(self, x, mean, cov, bib='numpy'):
        if bib == 'numpy':
            return -len(x)/2*np.log(2*np.pi) - 1/2*(np.log(np.linalg.det(cov)) + (x - mean).T @ np.linalg.inv(cov) @ (x - mean))
        else:
            return -len(x)/2*torch.log(torch.tensor(2*torch.pi)) - 1/2*(torch.log(torch.linalg.det(cov)) + (x - mean).T @ torch.inverse(cov) @ (x - mean))

    def log_likelihood(self, X, bib='numpy', dimension_reduction=False):
        """
        Calculate the log-likelihood of a set of data points given a Gaussian mixture model.

        Parameters:
            X (ndarray): An array of shape (n_samples, n_features) containing the data points.
            bib (str, optional): The library to use for the mathematical operations. Defaults to 'numpy'.

        Returns:
            float: The log-likelihood of the data points under the Gaussian mixture model.
        """
        if dimension_reduction:
            X = torch.from_numpy(self.ipca.transform(X)).float()

        # Initialize an array to store the log probability densities
        log_pdf = np.zeros((X.shape[0], len(self.weights)))

        # Calculate the log probability density for each data point and mixture component
        for i in range(X.shape[0]):
            for j in range(len(self.weights)):
                log_pdf[i, j] = np.log(self.weights[j]) + self.log_normal_pdf(X[i], self.means[j], self.cov[j], bib)
                
        # Sum log likelihoods of each data point
        return np.sum(logsumexp(log_pdf, axis=1))

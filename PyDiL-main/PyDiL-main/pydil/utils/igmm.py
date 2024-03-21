from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import copy
import torch
from scipy.special import logsumexp


class DynamicParameter(object):
    def __init__(self, *args, **conf):
        self.default_params()
        if len(args) > 1:
            raise ValueError
        elif len(args) == 1:
            if isinstance(args[0], float):
                self.values = args[0]
        elif len(conf.keys())>0:
            self.conf['is_constant'] = False
            for key in conf.keys():
                self.conf[key] = conf[key]
            if self.conf['function'] == 'log':
                self.log_evolution()
            elif self.conf['function'] == 'linear':
                self.linear_evolution()
        else:
            self.default_params()

    def log_evolution(self):
        init_val = self.conf['init']
        end_val = self.conf['end']
        steps = self.conf['steps']
        self.values = np.logspace(np.log(init_val), np.log(end_val), num=steps, base=np.exp(1))
        self.idx = -1
        self.conf['max_idx'] = steps - 1

    def linear_evolution(self):
        init_val = self.conf['init']
        end_val = self.conf['end']
        steps = self.conf['steps']
        self.values = np.linspace(init_val, end_val, num=steps)
        self.idx = -1
        self.conf['max_idx'] = steps - 1

    def default_params(self):
        self.conf = {'is_constant':True}
        self.values = 0.05

    def get_value(self):
        if self.conf['is_constant']:
            return self.values
        else:
            self.idx += 1
            if self.idx >self.conf['max_idx']:
                self.conf['is_constant'] = True
                self.values = self.values[-1]
                return self.get_value()
            return self.values[self.idx]

    def __print__(self):
        return str(self.value())

class IGMM(GMM):
    def __init__(self, min_components=3,
                 max_step_components=30,
                 max_components=60,
                 a_split=0.8,
                 forgetting_factor=DynamicParameter(0.05),
                 x_dims=None,
                 y_dims=None,
                 plot=False, plot_dims=[0, 1],
                 batch_size=1000):
        
        self.batch_size = batch_size

        GMM.__init__(self, n_components=min_components,
                     covariance_type='full')

        if isinstance(forgetting_factor, float):
            forgetting_factor = DynamicParameter(forgetting_factor)

        self.params = {'init_components': min_components,
                       'max_step_components': max_step_components,
                       'max_components': max_components,
                       'a_split': a_split,
                       'plot': plot,
                       'plot_dims': plot_dims,
                       'forgetting_factor': forgetting_factor,
                       'x_dims': x_dims,
                       'y_dims': y_dims,
                       'infer_fixed': False}

        if x_dims is not None and y_dims is not None:
            self.params['infer_fixed'] = True

        self.type='IGMM'
        self.initialized=False

    def fit(self, data):
        if self.initialized:
            ff_tmp = self.params['forgetting_factor'].get_value()

            self.short_term_model = IGMM(min_components=self.params['init_components'])
            self.short_term_model.get_best_gmm(data, lims=[1, self.params['max_step_components']])
            self.short_term_model.weights_ = ff_tmp * self.short_term_model.weights_
            self.weights_ = (1 - ff_tmp) * self.weights_

            gmm_new = self.merge_GMM(self.short_term_model)
            self.add_GMM(gmm_new)

            self.weights_=self.weights_/sum(self.weights_) #Regularization

        else:
            self.get_best_gmm(data, lims=[self.params['init_components'], self.params['init_components']])
            self.short_term_model = GMM(self.n_components)
            self.initialized = True

    def get_best_gmm(self, data, lims=[1, 10]):
        lowest_bic = np.infty

        n_components_range = range(lims[0], lims[1] + 1, 1)
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM, beware for cases where te model is not found in any case
            gmm = GMM(n_components=n_components,
                          covariance_type='full')
            gmm.fit(data)
            bic = gmm.bic(data)

            if bic < lowest_bic:
                lowest_bic = bic
                best_gmm = gmm

        self.weights_ = best_gmm.weights_
        self.covariances_ = best_gmm.covariances_
        self.means_ = best_gmm.means_
        self.n_components = best_gmm.n_components

    def get_bic(self, data):
        return self.bic(data)

    def merge_GMM(self, gmm2):
        while self.n_components + gmm2.n_components > self.params['max_components']:
            # Selecting high related Gaussians to be merged
            gmm1 = self
            similarity = self.get_distance_matrix(gmm1, gmm2)
            indices = np.unravel_index(similarity.argmin(), similarity.shape)
            gmm2 = self.merge_new_gauss_to_GMM(gmm2, indices[0],
                                        indices[1])
        return gmm2

    def merge_new_gauss_to_GMM(self, gmm2, index1, index2):
        gauss1 = {'covariance': self.covariances_[index1],
                  'mean': self.means_[index1],
                  'weight': self.weights_[index1]}
        gauss2 = {'covariance': gmm2.covariances_[index2],
                  'mean': gmm2.means_[index2],
                  'weight': gmm2.weights_[index2]}
        gauss = self.merge_gaussians(gauss1, gauss2)

        self.covariances_[index1] = gauss['covariance']
        self.means_[index1] = gauss['mean']
        self.weights_[index1] = gauss['weight']

        gmm2.covariances_ = np.delete(gmm2.covariances_, index2, 0)
        gmm2.means_ = np.delete(gmm2.means_, index2, 0)
        gmm2.weights_ = np.delete(gmm2.weights_, index2, 0)
        gmm2.n_components = gmm2.n_components - 1

        return gmm2
    
    def merge_gaussians(self, gauss1, gauss2):
        weight1 = gauss1['weight']
        covar1 = gauss1['covariance']
        mean1 = gauss1['mean']
        weight2 = gauss2['weight']
        covar2 = gauss2['covariance']
        mean2 = gauss2['mean']

        weight = weight1 + weight2
        f1 = weight1 / weight
        f2 = weight2 / weight
        mean = f1 * mean1 + f2 * mean2
        m1m2 = mean1 - mean2
        covar = f1 * covar1 + f2 * covar2 + f1 * f2 * np.outer(m1m2, m1m2)

        return {'covariance': covar, 'mean': mean, 'weight': weight}
    
    def add_GMM(self, gmm2):
        new_n_components = self.n_components + gmm2.weights_.shape[0]

        self.covariances_ = np.concatenate([self.covariances_, gmm2.covariances_], axis=0)
        self.means_ = np.concatenate([self.means_, gmm2.means_], axis=0)
        self.weights_ = np.concatenate([self.weights_, gmm2.weights_], axis=0)
        self.n_components = new_n_components
    
    def get_distance_matrix(self, gmm1, gmm2):
        n_comp_1 = gmm1.n_components
        n_comp_2 = gmm2.n_components
        similarity_matrix = np.full((n_comp_1, n_comp_2), np.inf)
        for i, (Mu, Sigma) in enumerate(zip(gmm1.means_, gmm1.covariances_)):
            Mu = torch.from_numpy(Mu).float()
            Sigma = torch.from_numpy(Sigma).float()
            for j, (Mu2, Sigma2) in enumerate(zip(gmm2.means_, gmm2.covariances_)):
                Mu2 = torch.from_numpy(Mu2).float()
                Sigma2 = torch.from_numpy(Sigma2).float()
                similarity_matrix[i, j] = self.bures_wasserstein_metric(Mu, Mu2, Sigma, Sigma2)
        return similarity_matrix

    def sqrtm(self, A, return_inv=False):
        D, V = torch.linalg.eig(A)

        A_sqrt = torch.mm(V, torch.mm(torch.diag(D.pow(1 / 2)), V.T)).real
        if return_inv:
            A_sqrt_neg = torch.mm(V, torch.mm(torch.diag(D.pow(-1 / 2)), V.T)).real
            return A_sqrt, A_sqrt_neg
        return A_sqrt

    def bures_wasserstein_metric(self, mean1, mean2, cov1, cov2):
        sqrt_cov1 = self.sqrtm(cov1)

        M = self.sqrtm(torch.mm(sqrt_cov1, torch.mm(cov2, sqrt_cov1)))
        bures_metric = torch.trace(cov1) + torch.trace(cov2) - 2 * torch.trace(M)

        return torch.sqrt(torch.dist(mean1, mean2, p=2) ** 2 + bures_metric)
       
    def log_normal_pdf(self, x, mean, cov, bib='numpy'):
        if bib == 'numpy':
            return -len(x)/2*np.log(2*np.pi) - 1/2*(np.log(np.linalg.det(cov)) + (x - mean).T @ np.linalg.inv(cov) @ (x - mean))
        else:
            return -len(x)/2*torch.log(torch.tensor(2*torch.pi)) - 1/2*(torch.log(torch.linalg.det(cov)) + (x - mean).T @ torch.inverse(cov) @ (x - mean))

    def score_samples(self, X, bib='numpy'):
        """
        Calculate the log-likelihood of a set of data points given a Gaussian mixture model.

        Parameters:
            X (ndarray): An array of shape (n_samples, n_features) containing the data points.
            bib (str, optional): The library to use for the mathematical operations. Defaults to 'numpy'.

        Returns:
            float: The log-likelihood of the data points under the Gaussian mixture model.
        """
        # Initialize an array to store the log probability densities
        log_pdf = np.zeros((X.shape[0], len(self.weights_)))

        # Calculate the log probability density for each data point and mixture component
        for i in range(X.shape[0]):
            for j in range(len(self.weights_)):
                log_pdf[i, j] = np.log(self.weights_[j]) + self.log_normal_pdf(X[i], self.means_[j], self.covariances_[j], bib)
                
        # Sum log likelihoods of each data point
        return logsumexp(log_pdf, axis=1)
    
    def sample(self, n_samples=None):
        n_samples = self.batch_size if n_samples is None else n_samples
        if np.sum(np.array(self.weights_)[:-1].astype(np.float64)) > 1:
            self.weights_ = [self.weights_[i]/1.000001 for i in range(len(self.weights_))]
        n_samples_comp = np.random.multinomial(n_samples, np.array(self.weights_))
        X = np.vstack(
                [
                    np.random.multivariate_normal(mean, covariance, int(sample))
                    for (mean, covariance, sample) in zip(
                        self.means_, self.covariances_, n_samples_comp
                    )
                ]
            )
        return torch.from_numpy(X).float(), None
    
    def get_GMM(self):
        return [[self.weights_[i],self.means_[i], self.covariances_[i]] for i in range(len(self.weights_))]
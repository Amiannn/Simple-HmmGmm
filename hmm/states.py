import os
import numpy  as np
import pandas as pd

from sklearn.mixture import GaussianMixture

class States():
    def __init__(self, length, mixtures):
        self.length            = length
        self.initials_log      = self._init_initials()
        self.transition_matrix = self._init_tranition()
        self.emissions         = self._init_emission(mixtures)
    
    def _init_initials(self):
        initials_log     = np.zeros(self.length)
        initials_log[1:] = -float('inf')
        return initials_log

    def _init_tranition(self):
        self_transition = 0.9
        transition = np.diag((1- self_transition) * np.ones(self.length - 1), 1) \
                   + np.diag(self_transition * np.ones(self.length))
        transition[-1, -1] = 1.0
        return transition

    def _init_emission(self, mixtures):
        emissions = [
            GaussianMixture(
                n_components=mixtures, 
                covariance_type='full', 
                init_params='kmeans', 
                max_iter=50,
            ) for _ in range(self.length)
        ]
        return emissions

    def __str__(self):
        columns = ['S{}'.format(i) for i in range(self.length)]
        rows    = ['S{}'.format(i) for i in range(self.length)]

        trans_str = pd.DataFrame(self.transition_matrix, columns=columns, index=rows).__str__()
        try:
            emiss_str = ''.join(
                [
                    '{}<S{}>{}\nmean:\n{}\n\ncov:\n{}\n\nweight:\n{}\n\n'.format(
                        '-'*10,
                        i,
                        '-'*10,
                        gmm.means_, 
                        gmm.covariances_, 
                        gmm.weights_
                    ) for i, gmm in enumerate(self.emissions)
                ]
            )
        except:
            emiss_str = ''.join(['[S{}] mean: null, cov: null, weight: null\n'.format(i) for i in range(self.length)])
        
        return '{}Transition Matrix{}\n\n{}\n\n{}Emissions{}\n\n{}\n'.format('='*5, '='*5, trans_str, '='*9, '='*9, emiss_str)     
        # return '{}Transition Matrix{}\n\n{}\n\n{}Emissions{}\n\n{}\n'.format('='*5, '='*5, trans_str, '='*9, '='*9, '=')     

    def save(self, path):
        pi_path    = os.path.join(path, 'initials')
        trans_path = os.path.join(path, 'transition')
        gmm_path   = os.path.join(path, 'emissions')

        # create folder
        for p in [pi_path, trans_path, gmm_path]: 
            if not os.path.isdir(p):
                os.makedirs(p)

        # save initials
        with open(os.path.join(pi_path, 'pi.npy'), 'wb') as fr:
            np.save(fr, self.initials_log)

        # save transitions
        with open(os.path.join(trans_path, 'a.npy'), 'wb') as fr:
            np.save(fr, self.transition_matrix)

        # save emissions
        for s in range(self.length):
            np.save(os.path.join(gmm_path, '{}.weights'.format(s)), self.emissions[s].weights_, allow_pickle=False)
            np.save(os.path.join(gmm_path, '{}.means'.format(s)), self.emissions[s].means_, allow_pickle=False)
            np.save(os.path.join(gmm_path, '{}.covariances'.format(s)), self.emissions[s].covariances_, allow_pickle=False)
    
    def load(self, path):
        pi_path    = os.path.join(path, 'initials')
        trans_path = os.path.join(path, 'transition')
        gmm_path   = os.path.join(path, 'emissions')

        # load initials
        with open(os.path.join(pi_path, 'pi.npy'), 'rb') as fr:
            self.reset_initials(np.load(fr))

        # load transistions
        with open(os.path.join(trans_path, 'a.npy'), 'rb') as fr:
            self.reset_transition(np.load(fr))

        # load emissions
        emissions = []
        for s in range(self.length):
            means   = np.load(os.path.join(gmm_path, '{}.means.npy'.format(s)))
            covars  = np.load(os.path.join(gmm_path, '{}.covariances.npy'.format(s)))
            weights = np.load(os.path.join(gmm_path, '{}.weights.npy'.format(s)))
            loaded_gmm = GaussianMixture(n_components = len(means), covariance_type='full')
            loaded_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covars))
            loaded_gmm.means_       = means
            loaded_gmm.covariances_ = covars
            loaded_gmm.weights_     = weights
            emissions.append(loaded_gmm)
        self.reset_emissions(emissions)

    def reset_initials(self, initials_log):
        self.initials_log = initials_log         

    def reset_transition(self, transition_matrix):
        self.transition_matrix = transition_matrix

    def reset_emissions(self, emissions):
        self.emissions = emissions

    def transition(self, state_A, state_B):
        return self.transition_matrix[state_A, state_B]

    def emission(self, state_A):
        return self.emissions[state_A]
""" dpp_mbd
DPP Minibatch Diversification for Pytorch DataLoaders
Zhang, Kjellstrom and Mandt: Determinantal Point Processes for Mini-Batch Diversification

class KDPP
    Contains k-dpp inference operations
class DPPSampler
    Extends Pytorch samplers, yielding a sequence of indices sampled by the k-dpp upon iteration
Usage:
    Datapoints should be mapped to Y = {1, ..., n} by the user
    Similarities between (i, j), i, j in Y in this set should be defined in a kernel called the L kernel
    Indices in Y are sampled by the DPP

A model trained on a balanced training set gives higher predictive accuracy. 
Sampling through a k-dpp aims to generate a biased, but balanced subsample.
The alteration to standard SGD is simple and the update step of the algorithm is changed to
x_{t+1} = x_{t} - p_t 1/k  \sum_{i \in B} \nabla l(x, y_i), B ~ k-DPP 

Positive situations for DM-SGD primarily include those datasets which are not balanced.
Negative include those where the dataset is already inherently balanced (ie MNIST)



"""

from typing import Iterator, Sequence
from torch.utils.data import Sampler
import numpy as np
from scipy.linalg import eigh

class KDPP:
    def __init__(self, L : np.ndarray, k : int) -> None:
        
        self.evals, self.evecs = eigh(L)

        self.k = k
        self.N = self.evals.shape[0]

    def elementary_symmetric_polynomial(self) -> np.ndarray:
        """ Eigenvalues evals, kth elementary symmetric polynomial.
        Returns the kth elementary symmetric polynomial
        """
        e = np.zeros(shape=(self.k + 1, self.N + 1))
        e[0, :] = 1
        e[1:self.k+1, 0] = 0

        for l in range(1, self.k + 1):
            for n in range(1, self.N + 1):
                e[l, n] = e[l, n-1] + self.evals[n-1] * e[l-1][n-1]
        
        #e = np.delete(e, 0, axis=(0))
        #e = np.delete(e, 0, axis=(1))
        
        return e
    
    def sample_k_mixture_components(self) -> list:
        """
        Compute the chosen mixture components for the k-DPP (phase 1 of sampling)
        Return an array J of numbers corresponding to which evecs are chosen.
        Kuszela and Taskar Algorithm 8
        """
        """ First compute the symmetric polynomials """
        e = self.elementary_symmetric_polynomial()
        # Output set
        J = []
        l = self.k
        
        for n in range(self.N, 1, -1):
            # Chosen all evecs
            if l == 0:
                break
            
            # Compute the marginals
            if n == l:
                marginal = 1
            else:
                marginal = self.evals[n-1] * (e[l-1,n-1] / e[l,n])
            
            # Note we use n - 1 to account for the 0-indexing
            if np.random.uniform(0,1) < marginal:
                J.append(n - 1)
                l = l - 1

        return J
    
    def sample_exact_k(self) -> list:
        """ evals, evecs - eigendecomposition of L kernel.
        Samples an L DPP according to Algorithm 1 in Kuszela and Taskar
        (spectral decomposition method)
        """
        
        """ Sampling phase 1
        Calculate the 'inclusion probabilities'/'mixture weight'
        for each eigenvalue by
        mixture weight = lambda_i / (lambda_i + 1)
        Construct corresponding set of eigenvectors by restricting the
        set of eigenvectors to the set indexed by the chosen eigenvalues 
        """
        # Define the ground set size
        N = self.evals.shape[0]
        # Construct corresponding set of eigenvectors
        J = self.sample_k_mixture_components()
        V = self.evecs[:, J]
        dim_V = V.shape[1]
        
        # Hold output
        Y = []
        
        """ Sampling phase 2
        We calculate the probabilities of selecting each item from the ground
        set as in Kuszela and Taskar Algorithm 1, that is,
        P(i) = \sum_{v \in V} (v^T e_i)^2
        Choose some i. (item)
        Now find a vector to eliminate. (index) 
        To do this, find the index of
        some vector in V which has a nonzero component e_i.
        Obtain the subspace perpendicular to e_i and orthogonalise.
        Note that QR decomposition is more numerically stable than 
        Gram-Schmidt for this purpose, but the effect achieved is identical.
        """
        
        for i in range(dim_V-1, -1, -1):
            P = np.sum(V**2, axis = 1)
            P = P / np.sum(P)
            
            # Choose an element from the ground set according 
            # to the probabilities
            item = np.random.choice(range(N), p=P)
            # row_ids is a scalar
            
            # Get the first nonzero vector 
            # First vector which has a nonzero element in that row
            # We will find subspace orthogonal to this vector
            index = np.nonzero(V[item])[0][0]
            # Note that axis 0 of the V array corresponds to the rows
            # So we can ensure we will not get out of bounds error
            
            Y.append(item)

            # update V
            # V_j is the vector we don't like
            V_j = np.copy(V[:,index])
            V = V - np.outer(V_j, 
                            V[item]/V_j[item])
            V[:,index] = V[:,i]
            V = V[:,:i]

            # Orthogonalise by using qr decomposition
            V, _ = np.linalg.qr(V)

        return Y


class DPPSampler(Sampler[int]):
    """Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    indices : Sequence[int]

    def __init__(self, indices : Sequence[int], k_dpp : KDPP) -> None:
        # indices might actually be a redundant variable here since indices is encoded within the k-dpp
        self.indices = indices
        self.k_dpp = k_dpp

    def __iter__(self) -> Iterator[int]:
        # Obtain a relevant subset of the indices
        dpp_sample = self.k_dpp.sample_exact_k()
        # Yield it
        for i in dpp_sample:
            yield i

    def __len__(self) -> int:
        return len(self.indices)
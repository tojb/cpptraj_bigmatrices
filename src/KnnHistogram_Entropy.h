#ifndef KNNHISTOGRAM_ENTROPY_H
#define KNNHISTOGRAM_ENTROPY_H

#include <vector>
#include <cstddef>


//
//quick KNN information-theoretic entropy estimator
//after Numata et el.
//
//Josh Berryman 28.04.2026 josh.berryman@uni.lu
//
// Note: don't get excited, knn is not very effective
// up to higher dimensions. The present code is useful
// as a small part of a more complex entropy estimator, 
// see Analysis_Entropy*
//
//


/*
  Digamma for positive integer arguments.
  For n >= 1:
    psi(n) = H_{n-1} - gamma
  Used for exact finite-sample correction in kNN entropy.
*/
double digamma_integer(std::size_t n);

/*
  kNN entropy contribution for an n × d data block.

  Computes:
      log V_d + d <log eps_k>

  where eps_k is the distance to the k-th nearest neighbour.

  Arguments:
    X  : pointer to row-major data [n × d]
    n  : number of samples (frames)
    d  : block dimensionality
    k  : neighbour rank (typically 3–6)

  Returns:
    The data-dependent part of the Kozachenko–Leonenko entropy estimator.
*/
double knn_entropy_block(
  const double* X,
  std::size_t   n,
  std::size_t   d,
  std::size_t   k
);

/*
  Joint non-Gaussian negentropy for a block of DOFs.

  Computes:
      J = H_gauss - H_knn

  where:
    - H_gauss is the Gaussian entropy with the same covariance
    - H_knn is the kNN entropy estimate

  Arguments:
    Xc         : pointer to centered trajectory data [n_frames × p]
    n_frames   : number of samples
    p          : full dimensionality of Xc
    block_dofs : indices of DOFs belonging to this block
    k          : neighbour rank (default chosen by caller)

*/
double block_negentropy_knn(
  const double*              Xc,
  std::size_t                n_frames,
  std::size_t                p,
  const std::vector<std::size_t>& block_dofs,
  std::size_t                k
);

#endif // KNNHISTOGRAM_ENTROPY_H

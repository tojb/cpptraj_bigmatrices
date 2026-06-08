#ifndef ENTROPY_FT_H
#define ENTROPY_FT_H

#include <vector>
#include <cstddef>

/*
  Fourier-based entropy analysis for PCA-projected coordinates.

  Input:
    Xc        : pointer to centred, mass-weighted trajectory data [n_frames × d]
    n_frames  : number of samples
    D         : total dimensionality
    k         : number of nearest neighbours to treat... should be greater than dimensionality of the block?
*/


struct MatrixView; //forward declaration of structured type (see ChowLiuTree.h)

double fft_knn_block(
  const MatrixView&        Cfull,   // global covariance (mass-weighted)
  const std::vector<std::size_t>&  dofs,
  const double*            Xc,      // [n_frames × D_total]
  std::size_t              n_frames,
  std::size_t              D_total,
  std::size_t              k_knn );





#endif

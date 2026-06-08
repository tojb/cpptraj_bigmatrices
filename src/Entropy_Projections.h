#ifndef ENTROPY_PROJECTIONS_H
#define ENTROPY_PROJECTIONS_H

//function to do stochastic (cheap and cheerful) dimensionality reduction via Johnson-Lindenstrauss
void downproject_JL(
    const double*              X,        // full [nframes × D_full]
    size_t                     nframes,  //number of frames
    const std::vector<size_t>& dof,      //vector of indices to downproject
    size_t                     d_proj,   //target dimensionality
    std::vector<double>&       Y,        // [nframes × d_proj]
    uint64_t                   seed );   //RNG seed.

///Function to do Johnson-Lindenstrauss dimensionality reduction.
/// This version operates on a merges pair of DOFS, and also
//  returns the projections on the *same* random vectors of the
//  two individual blocks that were merged.
//
void downproject_and_split_JL(
    const double*              X,        // full trajectory [nframes × D_full]
    size_t                     nframes,  // number of frames
    size_t                     D_full,   // number of dof of whole trajectory
    const std::vector<size_t>& dofA,     // DOF block A
    const std::vector<size_t>& dofB,     // DOF block B
    size_t                     d_proj,   // reduce dimensionality of subblock from dof to d_proj
    std::vector<double>&       YA,       // output block A     [nframes x d_proj]
    std::vector<double>&       YB,       // output block B     [nframes x d_proj]
    std::vector<double>&       YAB,      // output joint block [nframes x d_proj]
    uint64_t                   seed  );  // rng seed

//
//
// Get the Schlitter entropy of the down-projected block *without* recalculating C
//
//
double jl_schlitter_block(
  const MatrixView&          C_full,
  const std::vector<size_t>& dof,
  size_t                     d_proj,
  uint64_t                   seed);




#endif

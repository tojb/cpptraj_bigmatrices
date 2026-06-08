
#include <vector>
#include <random>
#include <cmath>

#include "Analysis_EntropyHD.h"
#include "ChowLiuTree_Entropy.h"
#include "Entropy_Projections.h"
 

//define a cheap hash function to supplement or replace (expensive) RNG calls.
static inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL; //use my credit card number (XORed with a random number) as the salt.
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

//map the hashed output to (0,1)
inline double hash_to_uniform(uint64_t h) {
  // map to (0,1)
  return ((h >> 11) + 1.0) * (1.0 / (1ULL << 53));
}

////////////////
// Get a randomish number as a function of indices i and k
// this is deterministic pseudo-random Gaussian-ish RNG
// delivered cheaply, not good enough for MC or thermostat applications
// but plenty for JL projection of MD trajectories.
//
// Having deterministic random numbers for a given (i, k, seed )
// is necessary for this application.
//
/////
inline double R_ik(size_t i, size_t k, uint64_t seed, double scale)
{
  uint64_t h1 = splitmix64(seed ^ (i * 0x9e3779b97f4a7c15ULL + k));
  uint64_t h2 = splitmix64(seed ^ (k * 0x94d049bb133111ebULL + i));
  double   u1 = hash_to_uniform(h1);
  double   u2 = hash_to_uniform(h2);

  // Box–Muller
  double r = std::sqrt(-2.0 * std::log(u1));
  double theta = 2.0 * M_PI * u2;
  return scale * r * std::cos(theta); 
}

///function to do Johnson-Lindenstrauss dimensionality reduction.
void downproject_JL(
    const double*              X,        // full trajectory [nframes × D_full]
    size_t                     nframes,  // number of frames
    size_t                     D_full,   // number of dof of whole trajectory
    const std::vector<size_t>& dof,      // block indices
    size_t                     d_proj,   // reduce dimensionality of subblock from dof to d_proj
    std::vector<double>&       Y,        // output [nframes × d_proj]
    uint64_t                   seed  )   // rng seed
{
  const size_t    D     = dof.size();
  const double    scale = 1./std::sqrt((double)d_proj);

  Y.assign(nframes * d_proj, 0.0);


  // Perform projection: Y = X_block * R
  for (size_t f = 0; f < nframes; f++) { 
    double *y = &Y[ f * d_proj ]; //output frame start pointer   
    for (size_t i = 0; i < D; i++) {
      double xi = X[f * D_full + dof[i]]; //uncompressed frame start pointer for this dof.
      for (size_t k = 0; k < d_proj; k++) {

        /*****/
        double rik = R_ik(i, k, seed, scale); //get a random number 
        y[k] += xi * rik;                     //project.            
        /*****/

      }
    }
  }
}

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
    uint64_t                   seed  )   // rng seed
{
  const size_t    DA     = dofA.size();
  const size_t    DB     = dofB.size();
  const double    scale = 1./std::sqrt((double)d_proj);

  YA.assign(nframes * d_proj,  0.0);
  YA.assign(nframes * d_proj,  0.0);
  YAB.assign(nframes * d_proj, 0.0);

  // Perform projection: Y = X_block * R
  for (size_t f = 0; f < nframes; f++) {
    double *yA  = &YA[ f * d_proj ]; //output frame start pointer
    double *yB  = &YB[ f * d_proj ];
    double *yAB = &YAB[ f * d_proj ];

    // project A and also the A part of the joint block.
    for (size_t i = 0; i < DA; i++) {
      double xi = X[f * D_full + dofA[i]]; //uncompressed frame start pointer for this dof.
      for (size_t k = 0; k < d_proj; k++) {

        /*****/
        double rik = R_ik(i, k, seed, scale); //get a random number, repeatably for i,k.
        yA[k]  += xi * rik;                   //project.
	yAB[k] += xi * rik;
        /*****/

      }
    }
    // project B and also the B part of the joint block.
    for (size_t i = 0; i < DB; i++) {
      double xi = X[f * D_full + dofB[i]]; //uncompressed frame start pointer for this dof.
      size_t i_joint = i + DA;             //indexing appears continuous in the joint block
      for (size_t k = 0; k < d_proj; k++) {

        /*****/
        double rik = R_ik(i_joint, k, seed, scale); //get a random number
        yB[k]  += xi * rik;                   //project.
        yAB[k] += xi * rik;
        /*****/

      }
    }
  }
}

//get the Schlitter entropy of a down-projected block
//*without* recalculating C from scratch in the projected space.
double jl_schlitter_block(
  const MatrixView&          C_full,
  const std::vector<size_t>& dof,
  size_t                     d_proj,
  uint64_t                   seed)
{
  
  size_t D = dof.size();
  std::vector<double> Cproj(d_proj * d_proj, 0.0);

  //loop over existing covariance submatrix elements
  for (size_t i = 0; i < D; i++) {
    size_t ii = dof[i];
    for (size_t j = 0; j <= i; j++) {
      size_t jj = dof[j];
      double cij = C_full(ii, jj);

      //calculate contribution of c_ij in Cproj
      for (size_t k = 0; k < d_proj; k++) {

        double rik = R_ik(i, k, seed, 1.0); //repeatable RNG based on index.
        for (size_t l = 0; l <= k; l++) {
          double rjl = R_ik(j, l, seed, 1.0);

	  //accumulate to the projected covariance matrix
          Cproj[k * d_proj + l] += rik * cij * rjl;
        }
      }
    }
  }

  // symmetrise (fill upper triangle)
  for (size_t k = 0; k < d_proj; k++) {
    for (size_t l = 0; l < k; l++) {
      Cproj[l * d_proj + k] = Cproj[k * d_proj + l];
    }
  }
  MatrixView Cview{ Cproj.data(), d_proj };

  std::vector<size_t> dof_proj(d_proj);
  for (size_t k = 0; k < d_proj; k++)
    dof_proj[k] = k;

  return block_entropy_logdet(Cview, dof_proj, false, 1e-12);
}




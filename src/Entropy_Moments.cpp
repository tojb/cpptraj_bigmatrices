#include <vector>
#include <cmath>
#include <algorithm>

#include "Entropy_Moments.h"


//////////////////////////////////////////////////////////////
//
// Computes NEW non-Gaussian entropy corrections introduced by merging blocks A and B
// Assumes:
//   - internal H(A), H(B) already harvested, so individual blocks are now multivariate Gaussians.
//   - A.dofs and B.dofs are disjoint
// Returns:
//   - An entropy correction Sng_merge >= 0.
//
// Why are we doing this? Because Schlitter entropy is a good bound for systems which are
// Gaussian. To tighten the bound, we look at deviations from Gaussian.
//
// 1-dof deviations can come from a 1D Shannon entropy, but N-dimensional Shannon is hard.
//
// We find the N-body deviations in NlogN merges going up a tree
// constructed heuristically to harvest the most information at the lowest levels.
//
/////
double sng2_merge_cca(const double* Xc,
	              size_t        n_frames, 
		      size_t        p,
                      const         TreeNode& A, 
		      const         TreeNode& B,
                      size_t        max_modes)
{
  const size_t m = A.dofs.size();
  const size_t n = B.dofs.size();
  if (m == 0 || n == 0 || n_frames == 0) return 0.0;

  DataSet_MatrixDbl SigmaA, SigmaB, SigmaAB;
  SigmaA.AllocateHalf(m);
  SigmaB.AllocateHalf(n);
  SigmaAB.Allocate2D(m, n);

  /* Build second-moment blocks. */
  for (size_t f = 0; f < n_frames; ++f) {
    const size_t fp = f * p;
    AccumulateHalf(SigmaA, Xc, fp, A.dofs.data(), m);
    AccumulateHalf(SigmaB, Xc, fp, B.dofs.data(), n);
    AccumulateRect(SigmaAB, Xc, fp, A.dofs.data(), m, B.dofs.data(), n);
  }

  NormalizeSecondMoments(SigmaA, SigmaB, SigmaAB, n_frames);

  /* Eigenstuff used to construct whiteners. */
  DataSet_Modes modesA, modesB, modesCCA;
  modesA.CalcEigen_General(SigmaA);
  modesB.CalcEigen_General(SigmaB);

  /* Truncated whiteners:  drop collapsed dimensions from the analysis.*/
  DataSet_MatrixDbl WA, WB;
  const size_t mEff = BuildWhitenerTrunc(WA, modesA, m);
  const size_t nEff = BuildWhitenerTrunc(WB, modesB, n);
  if (mEff == 0 || nEff == 0) return 0.0;

  DataSet_MatrixDbl Cwh;
  BuildWhitenedCrossCov(Cwh, WA, SigmaAB, WB, mEff, nEff);

  DataSet_MatrixDbl M;
  BuildCCAMatrixHalf(M, Cwh, mEff, nEff);

  if (max_modes == 0) max_modes = mEff;
  if (max_modes > mEff) max_modes = mEff;
  modesCCA.CalcEigen(M, max_modes);

  /* Validity/stability checks for the CCA eigenvalues. */
  double rho2sum = 0.0, maxRho2 = 0.0;
  Rho2Stats(modesCCA, rho2sum, maxRho2);

  if (maxRho2 > 1.0 + 1e-6) return 0.0;

  const size_t k = modesCCA.Size();
  if (k == 0) return 0.0;

  /* Fourth-moment invariant in whitened coordinates. */
  const double mean_r4 = MeanR4Product(WA, mEff, WB, nEff, Xc, n_frames, p,
                                       A.dofs.data(), m, B.dofs.data(), n);

  const double gauss = double(k) + 2.0 * rho2sum;
  const double kappa = mean_r4 - gauss;
  if (kappa <= 0.0) return 0.0;

  return kappa / 48.0;
}






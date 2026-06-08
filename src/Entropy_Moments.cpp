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
double sng2_merge_cca(const double*     Xc,
                      size_t            n_frames,
                      size_t            p,
                      TreeNode&         A,
                      TreeNode&         B,
                      size_t            max_modes,
                      MomentsWorkspace& ws)
{
  const size_t m = A.dofs.size();
  const size_t n = B.dofs.size();
  if (m == 0 || n == 0 || n_frames == 0) return 0.0;

  DataSet_MatrixDbl& SigmaA  = ws.SigmaA;
  DataSet_MatrixDbl& SigmaB  = ws.SigmaB;
  DataSet_MatrixDbl& SigmaAB = ws.SigmaAB;

  /* Build second-moment blocks. */
  for (size_t f = 0; f < n_frames; ++f) {
    const size_t fp = f * p;
    AccumulateHalf(SigmaA, Xc, fp, A.dofs.data(), m);
    AccumulateHalf(SigmaB, Xc, fp, B.dofs.data(), n);
    AccumulateRect(SigmaAB, Xc, fp, A.dofs.data(), m, B.dofs.data(), n);
  }

  NormalizeSecondMoments(SigmaA, SigmaB, SigmaAB, n_frames);

  /* Eigenstuff used to construct whiteners. */
  DataSet_Modes& modesA   = ws.modesA;
  DataSet_Modes& modesB   = ws.modesB;
  DataSet_Modes& modesCCA = ws.modesCCA;

  modesA.CalcEigen_General(SigmaA);
  modesB.CalcEigen_General(SigmaB);

  /* Truncated whiteners:  drop collapsed dimensions from the analysis.*/
  
  const size_t mEff = BuildWhitenerTrunc(m, ws.modesA, ws.WA, ws);
  const size_t nEff = BuildWhitenerTrunc(n, ws.modesB, ws.WB, ws);
  if (mEff == 0 || nEff == 0) return 0.0;
  
  BuildWhitenedCrossCov(mEff, nEff, ws);
  
  BuildCCAMatrixHalf(ws, mEff, nEff);

  if (max_modes == 0) max_modes = mEff;
  if (max_modes > mEff) max_modes = mEff;
 
  modesCCA.CalcEigen(ws.Mhalf, max_modes);

  /* Validity/stability checks for the CCA eigenvalues. */
  double rho2sum = 0.0, maxRho2 = 0.0;
  Rho2Stats(modesCCA, rho2sum, maxRho2);

  if (maxRho2 > 1.0 + 1e-6) return 0.0;

  const size_t k = modesCCA.Size();
  if (k == 0) return 0.0;

  /* Fourth-moment invariant in whitened coordinates. */
  // ...first, get second moments
  const double mean_r2A = MeanR2Block(ws.WA, mEff, Xc, n_frames, p, A.dofs.data(), m);
  const double mean_r2B = MeanR2Block(ws.WB, nEff, Xc, n_frames, p, B.dofs.data(), n);

  // Compute raw fourth moment
  const double mean_r4 = MeanR4Product(ws.WA, mEff, ws.WB, nEff, Xc, n_frames, p, A.dofs.data(), m, B.dofs.data(), n);

  // Convert to connected (covariance) estimator correction relative to second-order terms.
  const double connected_r4 = mean_r4 - mean_r2A * mean_r2B;

  // Subtract Gaussian expectation also
  const double kEff = static_cast<double>(modesCCA.Size());
  if (kEff == 0.0) return 0.0;
  const double kappa = connected_r4 - (2.0 * rho2sum / kEff);

  if (!std::isfinite(kappa)) return 0.0;
//  if (kappa <= 0.0) return 0.0; don't introduce bias by clipping (non-physical) neg values.
  
  return kappa / 48.0;
}






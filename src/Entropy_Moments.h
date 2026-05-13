// Entropy_moments.h
#ifndef ENTROPY_MOMENTS_H
#define ENTROPY_MOMENTS_H

#include "DataSet.h"
#include "DataSet_Modes.h"
#include "DataSet_MatrixDbl.h"
#include "CpptrajStdio.h"

#include "ChowLiuTree_Entropy.h" //TreeNode

#include <cstddef>   // size_t

#include <vector>
#include <cmath>
#include <algorithm>


//export this function, to analyse a centred and rigid-body fitted
//trajectory Xc for deviations from Gaussian covariance.
double sng2_merge_cca(  const double* Xc,
	                size_t        n_frames,  
		        size_t        p,
		      	const         TreeNode& A,
		      	const         TreeNode& B,
		      	size_t        max_modes ); 


//inlined hlper function to find the max finite eigenvalue
static inline double MaxFiniteEigenvalue(const DataSet_Modes& modes, size_t nOrig)
{
  double max_ev = 0.0;
  for (size_t i = 0; i < nOrig; i++) {
    const double ev = modes.Eigenvalue(i);
    if (std::isfinite(ev) && ev > max_ev) max_ev = ev;
  }
  return max_ev;
}



/* 
     This inlined helper function builds a whitening matrix W that maps an input vector of length nOrig
     into a reduced “whitened” vector space. The reduction happens by discarding modes
     whose eigenvalues are too small to safely invert.

     The matrix returned is rectangular: W has nEff rows and nOrig columns, where nEff is
     the number of retained modes.
*/
static inline size_t BuildWhitenerTrunc( DataSet_MatrixDbl&   W,
                                         const DataSet_Modes& modes,
                                         size_t               nOrig )
{
  // Fixed stability settings: avoid inverting very small eigenvalues.
  constexpr double REL_EPS   = 1e-6;
  constexpr double MAX_COND  = 1e8;
  constexpr double MAX_SCALE = 1e4;
  constexpr double HARD_FLOOR = 1e-30;

  const double max_ev = MaxFiniteEigenvalue(modes, nOrig);

  // No usable spectrum means no stable whitening transform.
  if (!(max_ev > 0.0) || !std::isfinite(max_ev)) {
    W.Allocate2D(0, nOrig);
    return 0;
  }

  // Combine absolute tolerance, relative tolerance, a condition-number cap, and a scale cap.
  double thr = HARD_FLOOR;
  thr = std::max(thr, REL_EPS * max_ev);
  thr = std::max(thr, max_ev / MAX_COND);
  thr = std::max(thr, 1.0 / (MAX_SCALE * MAX_SCALE));

  // Select eigenmodes that are safe to invert.
  std::vector<size_t> keep;
  keep.reserve(nOrig);
  for (size_t i = 0; i < nOrig; i++) {
    const double ev = modes.Eigenvalue(i);
    if (!std::isfinite(ev) || ev <= 0.0) continue;
    if (ev < thr) continue;

    const double scale = 1.0 / std::sqrt(ev);
    if (!std::isfinite(scale) || scale > MAX_SCALE) continue;

    keep.push_back(i);
  }

  // Build the rectangular whitener: one row per retained eigenmode.
  const size_t nEff = keep.size();
  W.Allocate2D(nEff, nOrig);
  for (size_t ii = 0; ii < nEff; ii++) {
    const size_t i = keep[ii];
    const double ev = modes.Eigenvalue(i);
    const double scale = 1.0 / std::sqrt(ev);
    const double* ei = modes.Eigenvector(i);
    for (size_t j = 0; j < nOrig; j++) {
      const double v = scale * ei[j];
      W.SetElement(ii, j, std::isfinite(v) ? v : 0.0);
    }
  }

  return nEff;
}

//accumulator loop packed into an inline function for convenience
static inline void AccumulateHalf(DataSet_MatrixDbl& S,
                                  const double*      Xc,
                                  size_t             fp,
                                  const size_t*      idx,
                                  size_t             d)
{
  for (size_t i = 0; i < d; ++i) {
    const double xi = Xc[fp + idx[i]];
    for (size_t j = 0; j <= i; ++j) {
      S.UpdateElement(i, j, xi * Xc[fp + idx[j]]);
    }
  }
}

//accumulator loop packed into an inline function for convenience
static inline void AccumulateRect(DataSet_MatrixDbl& S,
                                  const double*      Xc,
                                  size_t             fp,
                                  const size_t*      Ad,
                                  size_t             m,
                                  const size_t*      Bd,
                                  size_t             n)
{
  for (size_t i = 0; i < m; ++i) {
    const double xi = Xc[fp + Ad[i]];
    for (size_t j = 0; j < n; ++j) {
      S.UpdateElement(i, j, xi * Xc[fp + Bd[j]]);
    }
  }
}

//inline helper to normalise three matrices by the same factor
static inline void NormalizeSecondMoments(DataSet_MatrixDbl& SigmaA,
                                          DataSet_MatrixDbl& SigmaB,
                                          DataSet_MatrixDbl& SigmaAB,
                                          size_t             n_frames)
{
  const double invN = (n_frames > 0) ? (1.0 / double(n_frames)) : 0.0;
  SigmaA.Normalize(invN);
  SigmaB.Normalize(invN);
  SigmaAB.Normalize(invN);
}

//inline helper to build a whitened cross covariance
static inline void BuildWhitenedCrossCov(DataSet_MatrixDbl&       Cwh,
                                         const DataSet_MatrixDbl& WA,
                                         const DataSet_MatrixDbl& SigmaAB,
                                         const DataSet_MatrixDbl& WB,
                                         size_t                  mEff,
                                         size_t                  nEff)
{
  Cwh.Allocate2D(mEff, nEff);

  DataSet_MatrixDbl tmp;
  tmp.Multiply(WA, SigmaAB);
  Cwh.Multiply_M2transpose(tmp, WB);
}

//inlined accumulator triple loop
static inline void BuildCCAMatrixHalf(DataSet_MatrixDbl& Mhalf,
                                      const DataSet_MatrixDbl& Cwh,
                                      size_t mEff,
                                      size_t nEff)
{
  Mhalf.AllocateHalf(mEff);
  for (size_t i = 0; i < mEff; i++)
    for (size_t j = 0; j <= i; j++)
      for (size_t k = 0; k < nEff; k++)
        Mhalf.UpdateElement(i, j, Cwh.GetElement(i, k) * Cwh.GetElement(j, k));
}

//inlined helper to get eigenvalue statistics
static inline void Rho2Stats(const DataSet_Modes& modesCCA,
                             double& rho2sum,
                             double& maxRho2)
{
  rho2sum = 0.0;
  maxRho2 = 0.0;
  for (size_t i = 0; i < modesCCA.Size(); i++) {
    const double r2 = modesCCA.Eigenvalue(i);
    rho2sum += r2;
    maxRho2 = std::max(maxRho2, r2);
  }
}

//inlined helper to get eigenvalue statistics
static inline double MeanR2Block(const DataSet_MatrixDbl& W,
                                 size_t                  nEff,
                                 const double*           Xc,
                                 size_t                  n_frames,
                                 size_t                  p,
                                 const size_t*           idx,
                                 size_t                  dOrig)
{
  double acc = 0.0;
  for (size_t f = 0; f < n_frames; f++) {
    double r2 = 0.0;
    for (size_t i = 0; i < nEff; i++) {
      double z = 0.0;
      for (size_t j = 0; j < dOrig; j++)
        z += W.GetElement(i, j) * Xc[f * p + idx[j]];
      r2 += z * z;
    }
    acc += r2;
  }
  return acc / double(n_frames);
}

//accumulator for r4
static inline double MeanR4Product(const DataSet_MatrixDbl& WA,
                                   size_t                  mEff,
                                   const DataSet_MatrixDbl& WB,
                                   size_t                  nEff,
                                   const double*           Xc,
                                   size_t                  n_frames,
                                   size_t                  p,
                                   const size_t*           Ad,
                                   size_t                  m,
                                   const size_t*           Bd,
                                   size_t                  n)
{
  double mean_r4 = 0.0;
  for (size_t f = 0; f < n_frames; f++) {
    double r2A = 0.0, r2B = 0.0;

    for (size_t i = 0; i < mEff; i++) {
      double z = 0.0;
      for (size_t j = 0; j < m; j++)
        z += WA.GetElement(i, j) * Xc[f * p + Ad[j]];
      r2A += z * z;
    }

    for (size_t i = 0; i < nEff; i++) {
      double z = 0.0;
      for (size_t j = 0; j < n; j++)
        z += WB.GetElement(i, j) * Xc[f * p + Bd[j]];
      r2B += z * z;
    }

    mean_r4 += r2A * r2B;
  }
  return mean_r4 / double(n_frames);
}

//frobernius nurm helper function
static inline double FrobeniusNorm(const DataSet_MatrixDbl& A, size_t r, size_t c)
{
  double acc = 0.0;
  for (size_t i = 0; i < r; i++)
    for (size_t j = 0; j < c; j++) {
      const double v = A.GetElement(i, j);
      acc += v * v;
    }
  return std::sqrt(acc);
}










#endif

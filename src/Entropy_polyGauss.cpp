#include "Entropy_polyGauss.h"

#include <cmath>
#include <limits>
#include <vector>
#include <numeric>
#include <algorithm>


/***********************************************************************
 *
 *  Polynomial-Gaussian Entropy Correction
 *
 *  This module implements a low-order polynomial-Gaussian (PolyGauss)
 *  model for estimating non-Gaussian entropy corrections in molecular
 *  dynamics trajectories.
 *
 *  The data X are modeled as a smooth deformation of a Gaussian
 *  reference:
 *
 *      x = y + g(y),    with y ~ N(0, Sigma)
 *
 *  where g(y) is a low-order polynomial (typically quadratic). The
 *  Gaussian entropy is given by the Schlitter / quasi-harmonic formula,
 *  and the non-Gaussian correction is computed from the expected
 *  log-determinant of the Jacobian of the deformation.
 *
 *  This approach is intended for low-dimensional blocks exhibiting
 *  coherent geometric deviations from Gaussianity (e.g. skewed,
 *  curved, or "banana-shaped" fluctuations). It is robust to temporal
 *  correlations in MD data, since fitting relies on global moments
 *  rather than local neighborhood estimates. Compare versus histogramming
 *  approaches and knn, which are inherently ground-up and 
 *  thefore are vulnerable to short-time correlation effects and oversampling.
 *
 *  The PolyGauss model is designed to be used hierarchically: simple
 *  Gaussian blocks remain unchanged, while blocks identified as
 *  strongly coupled (e.g. via mutual information or Chow-Liu trees)
 *  may be upgraded to polynomial-Gaussian form. Higher-dimensional
 *  residual corrections can be handled separately via cumulant or
 *  nonparametric methods.
 *
 *  Josh Berryman 29.04.2026 josh.berryman@uni.lu
 *
 *  References:
 *
 *    Schlitter, J.
 *      "Estimation of absolute and relative entropies of macromolecules
 *       using the covariance matrix"
 *      Chemical Physics Letters 215 (1993).
 *
 *    Karplus, M., and Kushick, J.N.
 *      "Method for estimating the configurational entropy of
 *       macromolecules"
 *      Macromolecules 14 (1981).
 *
 *    Press, W.H., et al.
 *      "Numerical Recipes, Section on nonlinear transformations and
 *       entropy"
 *
 ***********************************************************************/

#include "Entropy_polyGauss.h"

#include <cmath>
#include <vector>
#include <algorithm>

#include "DataSet_MatrixDbl.h"
#include "DataSet_Modes.h"



/*  linear algebra helpers */
static void zero(std::vector<double>& v) {
  std::fill(v.begin(), v.end(), 0.0);
}
static double dot(const double* a, const double* b, std::size_t d) {
  double s = 0.0;
  for (std::size_t i = 0; i < d; ++i) s += a[i] * b[i];
  return s;
}

/*  Frobenius norm of (C - I)  */

static double frob_norm_C_minus_I(const DataSet_MatrixDbl& C, size_t d)
{
  double acc = 0.0;
  for (size_t i = 0; i < d; i++) {
    for (size_t j = 0; j <= i; j++) {
      double cij = C.GetElement(i, j);
      if (i == j) cij -= 1.0;
      acc += cij * cij;
    }
  }
  return std::sqrt(acc);
}

/* ================================================================
   Main PolyGauss block fit
   ================================================================ */
bool fit_polyGauss_block(
  const double*        X,       // row-major [nframes × d]
  std::size_t          nframes,
  std::size_t          d,
  PolyGaussModel&      model
)
{
  model.d = d;
  model.mean.assign(d, 0.0);

  /*      Compute mean
   */

  for (size_t f = 0; f < nframes; f++)
    for (size_t i = 0; i < d; i++)
      model.mean[i] += X[f*d + i];

  for (double& v : model.mean)
    v /= double(nframes);

  /* Compute block covariance */
  DataSet_MatrixDbl C;
  C.AllocateHalf(d);
  for (size_t f = 0; f < nframes; f++) {
    for (size_t i = 0; i < d; i++) {
      double di = X[f*d + i] - model.mean[i];
      for (size_t j = 0; j <= i; j++) {
        double dj = X[f*d + j] - model.mean[j];
        C.UpdateElement(i, j, di * dj);
      }
    }
  }

  double invN = 1.0 / double(nframes);
  for (size_t i = 0; i < d; i++)
    for (size_t j = 0; j <= i; j++)
      C.SetElement(i, j, C.GetElement(i, j) * invN);

  /*  Symmetric eigendecomposition */
  DataSet_Modes modes;
  modes.CalcEigen( C, d );
  {
    /* Declare whitening matrices in a local scope */
    DataSet_MatrixDbl W, Winv;

    W.Allocate2D(d, d);
    Winv.Allocate2D(d, d);

    const double eps = 1.0e-12;

    for (size_t k = 0; k < d; k++) {
      double eval = modes.Eigenvalue(k);
      if ( eval < eps) eval = eps;

      double invs      = 1.0 / std::sqrt(eval);
      double s         = std::sqrt(eval);
      const double* ek = modes.Eigenvector(k);

      /* W = Lambda^{-1/2} * U^T */
      for (size_t j = 0; j < d; j++) {
        W.SetElement(k, j, invs * ek[j]);
      }

      /* Winv = U * Lambda^{1/2} */
      for (size_t j = 0; j < d; j++)
        Winv.SetElement(j, k, s * ek[j]);
    }   

    //and wastefully copy them out to a less-local scope...
    model.whitening.resize(d*d);
    model.dewhitening.resize(d*d);
    for (size_t i = 0; i < d; i++)
      for (size_t j = 0; j < d; j++) {
        model.whitening[i*d + j]   = W.GetElement(i, j);
        model.dewhitening[i*d + j] = Winv.GetElement(i, j);
      }
  }



  /* Transform data to whitened coordinates Y */
  std::vector<double> Y(nframes * d, 0.0);
  for (size_t f = 0; f < nframes; f++) {
    for (size_t i = 0; i < d; i++) {
      double s = 0.0;
      for (size_t j = 0; j < d; j++)
        s += model.whitening[i*d + j] *
             (X[f*d + j] - model.mean[j]);
      Y[f*d + i] = s;
    }
  }

  /***********
   Quadratic polynomial fit:
       g_i(y) = sum_{j <= k} Q_{i,j,k} y_j y_k

   This is the part that gets expensive in high dimensions,
   obliging a transition to a moments-based approach (CCA).
  ***********/
  model.Q.assign(d*d*d, 0.0);
  for (size_t f = 0; f < nframes; f++) {
    const double* y = &Y[f*d];
    for (size_t i = 0; i < d; i++)
      for (size_t j = 0; j < d; j++)
        for (size_t k = j; k < d; k++)
          model.Q[i*d*d + j*d + k] += y[i] * y[j] * y[k];
  }
  /**********/

  double scale = 1.0 / double(nframes);
  for (double& v : model.Q)
    v *= scale;

  /* Entropy correction via Jacobian expectation */
  model.entropy_correction =
    polyGauss_entropy_correction(model, Y.data(), nframes);

  return true;
}



double logdet_from_jacobian_cpptraj(const std::vector<double>& Jflat,
                                   size_t d)
{

  DataSet_MatrixDbl JTJ;
  JTJ.AllocateHalf(d); // symmetric matrix
  for (size_t i = 0; i < d; i++) {
    for (size_t j = 0; j <= i; j++) {
      double sum = 0.0;
      for (size_t k = 0; k < d; k++) {
        // (J^T J)_{ij} = sum_k J_{k,i} * J_{k,j}
        sum += Jflat[k*d + i] * Jflat[k*d + j];
      }
      JTJ.SetElement(i, j, sum);
    }
  }

  // Eigendecomposition of JTJ
  DataSet_Modes modes;
  if (modes.CalcEigen(JTJ, d) != 0)
    return -std::numeric_limits<double>::infinity();

  // log |det J| = 0.5 * sum log eigenvalues(J^T J)
  double logdet = 0.0;
  const double eps = 1.0e-12;

  for (size_t k = 0; k < d; k++) {
    double lambda = modes.Eigenvalue(k); // lambda = σ_k^2
    if (lambda < eps)
      return -std::numeric_limits<double>::infinity();
    logdet += 0.5 * std::log(lambda);
  }

  return logdet;
}


/**
 * Compute log|det J| via eigenvalues of symmetric Jacobian,
 * with a "failsoft" filter which scales down the value as the
 * calculuation starts to look less stable, based on the eigval
 * spectrum.
 *
 * Returns:
 *   finite logdet : a usable PolyGauss correction
 *   -infinity     : the Jacobian is near-singular: probably operating
 *                   at too high a dimension to use polyGauss entropy correction.
 *
 *
 * This code seems to be dogshit generated by GPT
 *
 */
double failsoft_logdet_from_jacobian(const std::vector<double>& Jflat,
                                     size_t                         d)
{
  DataSet_MatrixDbl Js;
  Js.AllocateHalf(d);   // symmetric storage, zero-initialized

  // Js = 0.5 * (J + J^T)
  for (size_t i = 0; i < d; i++) {
    for (size_t j = 0; j <= i; j++) {
      double v = 0.5 * (Jflat[i*d + j] + Jflat[j*d + i]);
      Js.SetElement(i, j, v);
    }
  }

  // Eigen decomposition
  DataSet_Modes modes;
  if (modes.CalcEigen(Js, 0) != 0)
    return -std::numeric_limits<double>::infinity();

  // Get log determinant
  double logdet = 0.0;
  const double eps = 1.0e-12;
  for (size_t k = 0; k < d; k++) {
    double lambda = modes.Eigenvalue(k);
    if (std::abs(lambda) < eps)
      return -std::numeric_limits<double>::infinity();
    logdet += std::log(std::abs(lambda));
  }

  // Failsoft:
  // Measure how far the Jacobian departs from the identity by
  // taking rms deviations of log deviations eigenvalues from 1.
  double D2 = 0.0;
  for (size_t k = 0; k < d; k++) {
    double ll = modes.Eigenvalue(k);
    ll = std::log( std::abs( ll ) );
    D2 += ll * ll;
  }
  double D = std::sqrt(D2 / d);

  // Characteristic deviation scale beyond which PolyGauss should
  // smoothly shut off: assume that large deviations indicate a broken calculation. 
  const double D0 = 3.0;

  // Controls how sharply the correction is suppressed. Higher values
  // give a steeper but still smooth cutoff.
  const double p = 2.0;

  // Smooth attenuation factor. This is guarantees that
  // small changes in the trajectory produce small changes in entropy.
  double w = std::exp( -std::pow(D / D0, p) );

  return logdet * w;
}



/* 
   Entropy correction: < log |det(I + d g / d y)| >
 */
double polyGauss_entropy_correction( const PolyGaussModel& model,
                                     const double*             Y, // whitened data [n × d]
                                     std::size_t          nframes )
{
  const size_t d   = model.d;
  double       acc = 0.0;

  for (size_t f = 0; f < nframes; f++) {
    /* Build Jacobian J = I + d g / d y */
    std::vector<double> J(d*d, 0.0);
    for (size_t i = 0; i < d; i++)
      J[i*d + i] = 1.0;
    const double* y = &Y[f*d];
    for (size_t i = 0; i < d; i++) {
      for (size_t j = 0; j < d; j++) {
        double s = 0.0;
        for (size_t k = 0; k < d; k++) {
          if (j == k)
            s += 2.0 * model.Q[i*d*d + j*d + k] * y[k];
          else
            s += model.Q[i*d*d + j*d + k] * y[k];
        }
        J[i*d + j] += s;
      }
    }


    // Apply a volume stabilisation factor:
    // 0 < alpha <= 1
    // alpha = 1   -> original PolyGauss
    // alpha < 1   -> damped deformation (recommended)
    const double alpha = 0.4; // just picked this at random.    
    for (size_t i = 0; i < d; i++) {
      for (size_t j = 0; j < d; j++) {
        if (i == j)
          J[i*d + j] = 1.0 + alpha * (J[i*d + j] - 1.0);
        else
          J[i*d + j] = alpha * J[i*d + j];
      }
    }



    /* log(det J) using cpptraj linear algebra,
     * with a catch for negative or zero eigenvalues  */
    //double logdet = failsoft_logdet_from_jacobian(J, d);
    double logdet = logdet_from_jacobian_cpptraj(J, d);
    if (!std::isfinite(logdet)) {
      continue; //this node is ill-conditioned, record zero correction
    }   

    acc += logdet;
  }

  return acc / double(nframes);
}


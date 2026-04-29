#ifndef ENTROPY_POLYGAUSS_H
#define ENTROPY_POLYGAUSS_H

#include <vector>
#include <cstddef>

/*
  Polynomial–Gaussian block model

  Models data X as:
      x = y + g(y),   y ~~ N(0, I)

  where g(y) is a low-order polynomial learned from data.
  Entropy correction is computed from the Jacobian of the map.
*/

struct PolyGaussModel {
  std::size_t d;                     // block dimension

  // Linear whitening transform
  std::vector<double> mean;          // size d
  std::vector<double> whitening;     // d × d row-major
  std::vector<double> dewhitening;   // d × d row-major

  // Quadratic correction:
  // g_i(y) = sum_{j,k} Q[i,j,k] * y_j * y_k   (j <= k)
  std::vector<double> Q;              // size d × d × d (packed symmetric in j,k)

  double entropy_correction;          // Delta S = E[log |det J|]
};

/*
  Fit a polynomial–Gaussian model to block data.

  Arguments:
    X          : row-major data [n × d]
    n          : number of frames
    d          : block dimension
    model      : output fitted model

  Requirements:
    - Data should be centered at equilibrium scale
    - Intended for low d (<= 6 recommended)
*/
bool fit_polyGauss_block(
  const double*       X,
  std::size_t         n,
  std::size_t         d,
  PolyGaussModel& model
);

/*
  Compute entropy correction for a fitted PolyGauss model.

  Returns:
    DeltaS = ⟨ log |det (I + dg/dy) | ⟩
*/
double polyGauss_entropy_correction(
  const PolyGaussModel& model,
  const double*         X,
  std::size_t           n
);

#endif // ENTROPY_POLYGAUSS_H

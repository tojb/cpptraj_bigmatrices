#include "KnnHistogram_Entropy.h"

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <omp.h>

namespace {

/* Euler–Mascheroni constant */
constexpr double EULER_GAMMA =
  0.577215664901532860606512090082402431;

/* Volume of the unit ball in R^d */
inline double unit_ball_volume(std::size_t d)
{
  return std::pow(M_PI, 0.5 * d) / std::tgamma(0.5 * d + 1.0);
}

/* Compute k-th neighbour distances for an n × d block */
void kth_neighbour_distances(
  const double* X,
  std::size_t   n,
  std::size_t   d,
  std::size_t   k,
  std::vector<double>& eps
)
{
  eps.resize(n);

#pragma omp parallel for schedule(dynamic, 8)
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<double> dists;
    dists.reserve(n - 1);

    const double* xi = X + i * d;

    for (std::size_t j = 0; j < n; ++j) {
      if (j == i) continue;

      const double* xj = X + j * d;
      double r2 = 0.0;

      for (std::size_t a = 0; a < d; ++a) {
        const double dx = xi[a] - xj[a];
        r2 += dx * dx;
      }

      dists.push_back(std::sqrt(r2));
    }

    std::nth_element(
      dists.begin(),
      dists.begin() + (k - 1),
      dists.end()
    );

    eps[i] = dists[k - 1];
  }
}

/* Empirical covariance determinant (small d, LU without pivot) */
double covariance_determinant(
  const double* X,
  std::size_t   n,
  std::size_t   d
)
{
  std::vector<double> C(d * d, 0.0);

#pragma omp parallel
  {
    std::vector<double> C_local(d * d, 0.0);

#pragma omp for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
      const double* xi = X + i * d;
      for (std::size_t a = 0; a < d; ++a)
        for (std::size_t b = 0; b < d; ++b)
          C_local[a * d + b] += xi[a] * xi[b];
    }

#pragma omp critical
    {
      for (std::size_t i = 0; i < d * d; ++i)
        C[i] += C_local[i];
    }
  }

  for (double& v : C)
    v /= double(n);

  /* LU determinant */
  double det = 1.0;
  std::vector<double> A = C;

  for (std::size_t i = 0; i < d; ++i) {
    const double piv = A[i * d + i];
    if (std::abs(piv) < 1e-14)
      return 0.0;

    det *= piv;

    for (std::size_t j = i + 1; j < d; ++j) {
      const double f = A[j * d + i] / piv;
      for (std::size_t k2 = i; k2 < d; ++k2)
        A[j * d + k2] -= f * A[i * d + k2];
    }
  }

  return det;
}

} // anonymous namespace


/* Exact digamma for integer arguments */
double digamma_integer(std::size_t n)
{
  if (n <= 1)
    return -EULER_GAMMA;

  double h = 0.0;
  for (std::size_t k = 1; k < n; ++k)
    h += 1.0 / double(k);

  return h - EULER_GAMMA;
}


/* kNN entropy contribution for a block */
double knn_entropy_block(
  const double* X,
  std::size_t   n,
  std::size_t   d,
  std::size_t   k
)
{
  if (d == 0 || n <= k)
    return 0.0;

  std::vector<double> eps;
  kth_neighbour_distances(X, n, d, k, eps);

  double avg_log_eps = 0.0;
  for (double e : eps)


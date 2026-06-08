
#include <random>
#include <vector>
#include <cmath>
#include <iostream>

#include "../Entropy_KnnHistogram.h"
#include "../Entropy_FFT.h"
#include "../ChowLiuTree_Entropy.h"

// generate correlated Gaussian samples using an autoregressive
// correlated process (AR1).
void generate_correlated_gaussian(
  std::vector<double>& X,
  size_t n,
  size_t d,
  double alpha)   // correlation strength (0=independent, ~1=strongly correlated)
{
  std::mt19937 rng(12345);
  std::normal_distribution<double> N(0.0, 1.0);

  X.resize(n * d);

  // initial point
  for (size_t a = 0; a < d; a++)
    X[a] = N(rng);

  for (size_t t = 1; t < n; t++) {
    for (size_t a = 0; a < d; a++) {
      double noise = N(rng);
      X[t*d + a] =
          alpha * X[(t-1)*d + a]
        + std::sqrt(1.0 - alpha*alpha) * noise;
    }
  }

  // centre each dimension
  for (size_t a = 0; a < d; a++) {
    double mean = 0.0;

    for (size_t t = 0; t < n; t++)
      mean += X[t*d + a];

    mean /= double(n);

    for (size_t t = 0; t < n; t++)
      X[t*d + a] -= mean;
  }

}

//quick local covariance builder
void build_covariance(
  const double* X,
  size_t n,
  size_t d,
  std::vector<double>& Cv   // dense row-major d×d
){
  Cv.assign(d*d, 0.0);

  // mean
  std::vector<double> mean(d, 0.0);
  for (size_t t = 0; t < n; ++t)
    for (size_t i = 0; i < d; ++i)
      mean[i] += X[t*d + i];

  for (size_t i = 0; i < d; ++i)
    mean[i] /= double(n);

  // covariance
  for (size_t t = 0; t < n; ++t)
  {
    for (size_t i = 0; i < d; ++i)
    {
      double di = X[t*d + i] - mean[i];
      for (size_t j = 0; j < d; ++j)
      {
        double dj = X[t*d + j] - mean[j];
        Cv[i*d + j] += di * dj;
      }
    }
  }

  double invN = 1.0 / double(n);
  for (size_t i = 0; i < d*d; ++i)
    Cv[i] *= invN;
}


//wrapper for the fft_knn code
double fft_knn_entropy_block_with_pca(
  const double* X,
  size_t n,
  size_t d,
  size_t k_knn ) {
  
  // build covariance and present as a MatrixView
  std::vector<double> Cv;
  build_covariance(X, n, d, Cv);
  MatrixView C{Cv.data(), d};

  // DOFs = full block
  std::vector<size_t> dofs(d);
  for (size_t i = 0; i < d; ++i) dofs[i] = int(i);

  //get the entropy estimate
  return fft_knn_block(C, dofs, X, n, d, k_knn);
}




//entropy of a Gaussian (unit variance, dimension d)
double gaussian_entropy(size_t d)
{
  return 0.5 * d * (1.0 + std::log(2.0 * M_PI));
}

//call KNN on synthetic data and compare to target
int main() {
  size_t n = 20000;
  size_t d = 4;
  size_t k = 5;

  std::vector<double> X;

  for (double alpha : {0.0, 0.5, 0.9, 0.99}) {

    //make the test data
    fprintf(stdout, "generating AR noise process... ");
    fflush(stdout);
    generate_correlated_gaussian(X, n, d, alpha);
    fprintf(stdout, "done\n");
    fflush(stdout);


    //get the correlation time
    fprintf(stdout, "estimating correlation time... ");
    fflush(stdout);
    double tau_est = estimate_tau_from_projection(X.data(), n, d);
    fprintf(stdout, " ... %e\n", tau_est);
    fflush(stdout);

    //reverse-engineer the entropy correction:
    double S_correlation = 0.;
    if ( tau_est > 1. ) 
      S_correlation = -0.5 * std::log(tau_est);

    fprintf(stdout, "knn entropy... ");
    fflush(stdout);
    double H_knn_corr = knn_entropy_block(X.data(), n, d, k); // knn version
    fprintf(stdout, " ... %e\n", H_knn_corr);
    fflush(stdout);
  

    fprintf(stdout, "decorrelated knn entropy... ");
    fflush(stdout);
    double H_fft      = fft_knn_entropy_block_with_pca(X.data(), n, d, k); //ftknn
    fprintf(stdout, " ... %e\n", H_fft);
    fflush(stdout);


    printf("\ncorrelation strength %.4f gives correlation time estimate: %e deltaS: %.2f\n", alpha, tau_est, S_correlation);
    printf("Knn estimate is : %.4f versus ftknn: %.4f versus true: %.4f\n", H_knn_corr,  H_fft, gaussian_entropy(d));

  }
}

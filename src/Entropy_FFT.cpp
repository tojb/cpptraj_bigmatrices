#include "Entropy_FFT.h"
#include "PubFFT.h"
#include "CpptrajStdio.h"
#include <vector>
#include <cmath>
#include <algorithm>

#include "Analysis_EntropyHD.h"
#include "ChowLiuTree_Entropy.h"
#include "DataSet_MatrixDbl.h"
#include "DataSet_Modes.h"
#include "Entropy_KnnHistogram.h"

// project centred and aligned mass-weighted data onto PCA modes
// Y: [n_frames × nmodes_eff]
// returns the number of projected modes nmodes_eff
static size_t pca_project( const double*          Xc,
                         std::size_t            n_frames,
                         std::size_t            D,
                         const DataSet_Modes&   modes,
                         std::vector<double>&   Y )
{

  //calculate effective number of non-collapsed modes
  double lambda_max = modes.Eigenvalue(0);
  std::size_t nmodes_eff = 0, nmodes_use;

  fprintf(stderr, "iterating over %zu modes\n", modes.Size());
  fflush(stderr);
  for (int i = 0; i < int(modes.Size()); ++i) {
    double ev = modes.Eigenvalue(i);
    fprintf(stderr, "%i : %e\n", i, ev);
    fflush(stderr);
    if (ev > lambda_max * 1e-8) 
       ++nmodes_eff;
    else
       break;
  }

  //assign workspace
  Y.assign(n_frames * nmodes_eff, 0.0);

  // Loop over frames
  for (std::size_t f = 0; f < n_frames; ++f) {

    //frame base pointers.
    const double* xf = Xc + f*D;
    double*       yf = Y.data() + f*nmodes_eff;

    // Loop over modes
    for (std::size_t m = 0; m < nmodes_eff; ++m) {

      // pointer to eigenvector m
      const double* evec = modes.Eigenvector(m);
      double        proj = 0.0;

      // dot product
      for (std::size_t i = 0; i < D; ++i)
        proj += xf[i] * evec[i];
      yf[m] = proj;
    }
  }
  return nmodes_eff;
}

// compute 1D entropy of a single mode (via FT or fallback)
static double mode_entropy_1d(
  const double* y,         // length n_frames
  std::size_t   n_frames
);

// build predictive model y_i ≈ f(previous modes)
// and return residual time series
static void model_residual(
  const double* Y,         // [n_frames × D]
  std::size_t   n_frames,
  std::size_t   D,
  std::size_t   target_mode,
  std::vector<double>& residual  // length n_frames
);


//get a subcovariance into a half-matrix 
static void pack_subcov_half(  const MatrixView&       Cfull,
                               const std::vector<size_t>& dofs,
                               DataSet_MatrixDbl&      Csub ) {

  const std::size_t d = dofs.size();
  Csub.AllocateHalf(d);
  for (std::size_t i = 0; i < d; ++i)
  {
    for (std::size_t j = 0; j <= i; ++j)
    {
      double val = Cfull(dofs[i], dofs[j]);
      Csub.SetElement(i, j, val);
    }
  }
}



//do an FFT over the pca'd mode projections
static void fft_samples(
  const std::vector<double>& Y,   // [n_frames × d_eff]
  std::size_t                n_frames,
  std::size_t                d_eff,
  std::vector<double>&       Z,          // output
  std::size_t&               d_fft,
  std::size_t&               n_samples
) {

  // FFT setup
  PubFFT fft;
  fft.SetupFFTforN((int)n_frames); //linear, 1D FFT.
  std::size_t fftN = fft.size();   //check success and get alloc'd size
  if ( fftN != n_frames ){
    mprinterr( "problems allocating fft: got %zu , wanted %zu\n", fftN, n_frames );
  }


  //fix dimensions
  std::size_t k_min  = 1;               // skip DC
  std::size_t n_freq = fftN / 2;
  n_samples          = (n_freq > k_min) ? (n_freq - k_min) : 0;
  d_fft              = 2 * d_eff;
  Z.resize(n_samples * d_fft);

  // Loop over PCA mode projections
  for (std::size_t i = 0; i < d_eff; ++i)
  {
    ComplexArray data(fftN);
    // mean removal: shouldn't need this, data was already centred. 
    // do it anyway just for the sake of clearing away numerical errors which might have accumulated.
    double mean = 0.0;
    for (std::size_t t = 0; t < n_frames; ++t)
      mean += Y[t*d_eff + i];
    mean /= double(n_frames);
    if ( mean * mean > 1e-12 ) {
       mprinterr("input data to PCA projection FFT was not centred. Mean value: %e\n", mean);
    }

    // load signal with zero-padding if needed.
    for (std::size_t t = 0; t < n_frames; ++t) {
      data[2*t]     = Y[t*d_eff + i] - mean;
      data[2*t + 1] = 0.0;
    }
    for (int k = 2*n_frames; k < 2*fftN; ++k)
      data[k] = 0.0;

    // FFT
    fft.Forward( data );

    // fill output samples across frequency bins
    for (std::size_t k = k_min; k < n_freq; ++k)
    {
      std::size_t ks = k - k_min;
      Z[ks*d_fft + 2*i    ] = data[2*k];
      Z[ks*d_fft + 2*i + 1] = data[2*k + 1];
    }
  }
}



// ================================================================
// Top-level FT entropy caller
// ================================================================

double fft_knn_block(
  const MatrixView&        Cfull,   // global covariance (mass-weighted)
  const std::vector<size_t>&  dofs,
  const double*            Xc,      // [n_frames × D_total]
  std::size_t              n_frames,
  std::size_t              D_total,
  std::size_t              k_knn ) {

  //unpack a subblock of the covariance and diagonalise it.
  DataSet_Modes modes;
  {
    DataSet_MatrixDbl Csub;
    pack_subcov_half( Cfull, dofs, Csub ); 
    modes.CalcEigen( Csub, dofs.size() );
  } //close scope for the subblock covariance.


  //project and take FFT of each projection
  std::vector<double> Z; //hold FFT'd projected DOF.
  std::size_t         n_samples, d_fft;
  {
    std::vector<double> Y; //time domain projected DOF.
    std::size_t d_eff = pca_project( Xc, n_frames, D_total, modes, Y );

    //FFT
    fft_samples(Y, n_frames, d_eff, Z, d_fft, n_samples);
  }

  //Make a knn joint entropy analysis in frequency space.
  //the ambition here is that time correlations (bane of knn) 
  //have been rotated away by the FFT
  double S_knn;
  mprintf("doing knn block entropy on fftd data n: %zu d: %zu k: %zu\n", n_samples, d_fft, k_knn);
  fflush(stdout);
  S_knn = knn_entropy_block( Z.data(), n_samples, d_fft, k_knn );

  return S_knn;

}


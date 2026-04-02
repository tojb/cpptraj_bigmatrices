// -----------------------------------------------------------------------------
// Analysis_EntropyHD.cpp
// cpptraj 'analysis entropy_hd' module
//
// Implements a Harris–Dryden bias-corrected Schlitter entropy estimator,
// with 95% CIs, OpenMP, and AMBER atom-mask support.
//
// Command example:
//   analysis entropy_hd dsout EntHD mask :1-22@P outfile entropy_vs_length.dat 
//                       details entropy_debug.txt c 1.0 kb 1.0 window 500
//
// Produces a user-named matrix dataset EntHD with columns:
//   n  S(n)  S0(n)  CIlo(n)  CIhi(n)
//
// References:
//   - Dryden, Kume, Le, Wood (Stat. inference for functions of covariance...,
//     Sections 4.1–4.2, Schlitter entropy and bias behavior).  
//   - Harris et al. (2001) empirical bias scaling for S(n). 
// -----------------------------------------------------------------------------
//
// This file is part of the AMBER software package, and is subject to the license terms
//
// Using pczdump’s Schlitter as the physical reference 
// (i.e., a sum of log⁡(1+α λi)\log(1+\alpha\,\lambda_i)log(1+αλi​) over covariance eigenvalues) 
// we here compute the same quantity without any eigensolve by switching to a frame‑space
// Gram formulation and a small Cholesky log‑det. 
// This is mathematically justified by Sylvester’s determinant identity: 
//  det⁡ ⁣(IP+α XcTXc)  =  det⁡ ⁣(IN+α XcXcT)\det\!\big(IP_ + \alpha\,X_c^{\mathsf T}X_c\big) \;=\;
//                \det\!\big(I_N + \alpha\,X_cX_c^{\mathsf T}\big)det(IP​+αXcT​Xc​)
//                =det(IN​+αXc​XcT​)
// so we can work in N×NN\times NN×N (frames) rather than P×PP\times PP×P (coordinates). 
// This delivers an eigensolve‑free Schlitter that is directly comparable to pczdump’s 
// ∑ilog⁡(1+α λi)\sum_i \log(1+\alpha\,\lambda_i)∑i​log(1+αλi​) 
// because both are exactly log⁡det⁡(I+α C)\log\det(I+\alpha\,C)logdet(I+αC).
//
// This is NOT directly comparable to the output of "diagmatrix thermo" because 
// that uses a different formula for the entropy per degree
// of freedom, based on the Quantum Harmonic Oscillator, which is unfortunately not easily
// expressed in a Gram formulation and thus does require an eigensolve.
//
// Deltas of entropy between different states or conditions should be
// comparable across both methods, however, since the different fudge factors 
// employed to avoid taking log of zero or negative eigenvalues should largely cancel out. 
//
// The "fudge factor" in the QHO formula is a constant added to the eigenvalues,
// with some physical justification, 
// while the "fudge factor" in the Schlitter formula is a constant multiplied by the covariance matrix, 
// with an unclear physical justification if any, so they are not directly comparable, 
// but both serve to regularize the entropy estimate.
//
// The advanced feature here is the Harris–Dryden bias correction, 
// which fits the observed S(n) to the empirical scaling form S(n) = S0 + m * n^{-a} 
// and extrapolates as n→∞ to get S0, which is a less biased estimate of the true configurational entropy.
//
// In production versions the classical Harris-Dryden is found not to actually work very well,
// especially for large systems where the number of frames needed to get a stable entropy estimate
// is actually quite a lot less than the number of frames needed to fully overdetermine the matrix,
// so cpptraj's Analysis_Regression framework has been extended to offer Harris-Dryden power law format
// as well as exponential saturation functions f(x)=A(1-exp(-Bx)) which seems to work better for larger
// systems / relatively shorter times.
//
// Caveat Emptor, Caveant Omnes
//
// 05/03/2026 Joshua T Berryman, josh.berryman@uni.lu.
//
//



#include <omp.h>
#include <cstring>              // std::memcpy()
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>

//debug print of heap state
#include <malloc.h>

#include "CpptrajStdio.h"      // mprintf(), mprinterr()
#include "DataSet.h"
#include "DataSet_1D.h"
#include "DataSetList.h"       // Dsets()
#include "DataFileList.h"
#include "DataSet_Coords.h"
#include "Topology.h"
#include "Frame.h"
#include "AtomMask.h"
#include "Analysis_EntropyHD.h"
#include "ThermoUtils.h"

static constexpr double kB   = 0.00198720425864083;   //kcal/mol/K
static constexpr double hbar = 0.0635078          ;   //kcal*ps/mol   

static int constructor_called = 0; // For testing whether constructor is called in unit tests. Remove when no longer needed.

// --------------------------------- constructor --------------------------------------
Analysis_EntropyHD::Analysis_EntropyHD() :
    P_( 0 ),
    nSelected_( 0 ),
    dsoutName_(""),
    coordsName_(""),
    maskString_(""),
    outfile_(nullptr),
    temp_(300.0), 
    pressure_(1.0),
    window_(500),
    do_jacobi_(false) ,
    mask_(), 
    Coords_(nullptr) ,
    AverageFrame_(),
    MassesPerAtom_(),
    AvgXYZ_(),
    outputDS_(),
    read_ptr_(0)
{
  ++constructor_called;
}

// -------------------------------- Help ---------------------------------------
void Analysis_EntropyHD::Help() const {
  mprintf("\nAnalysis entropy_hd : Bias-corrected Schlitter entropy (Harris–Dryden)\n");
  mprintf("Compute Schlitter entropy S(n) over increasing trajectory lengths and\n");
  mprintf("estimate the bias-corrected limit S0(n) with 95%% confidence intervals.\n\n");
  mprintf("USAGE:\n");
  mprintf("  dsout <name> [mask <atoms>] [outfile <file>] [details <file>]\n");
  mprintf("               [c <val>] [kb <val>] [temp <T>] [pressure <P>] [window <frames>] [jacobi]\n\n");
  mprintf("OPTIONS:\n");
  mprintf("  dsout    <name>   Name of output DataSet.\n");
  mprintf("  mask     <atoms>  AMBER atom mask to select coordinates (default: all atoms).\n");
  mprintf("  outfile  <file>   Write entropy data to file (default: stdout).\n");
  mprintf("  temp     <T>      Temperature in Kelvin (default 300.0).\n");
  mprintf("  pressure <P>      Pressure in atomspheres (default 1.0).\n");
  mprintf("  jacobi            Do a diagonalisation via Jacobi algoritm. Slow, but v. stable for long traj.\n");
  mprintf("  window  <frames>  Window size for n-grid; rows at n = window, 2*window, ... (default 500).\n\n");
  mprintf("OUTPUT DATASET COLUMNS:\n");
  mprintf("  1) n           Frames used (ending at final frame)\n");
  mprintf("  2) S(n)        Schlitter entropy: (1/kB) * log |I + c * Cov|\n");
  mprintf("  3) S0(n)       Bias-corrected entropy from LM fit: S(n) = S0 + m * n^{-a}\n");
  mprintf("  4) CIlo(n)     Lower 95%% confidence bound for S0(n)\n");
  mprintf("  5) CIhi(n)     Upper 95%% confidence bound for S0(n)\n\n");
  mprintf("EXAMPLE:\n");
  mprintf("  entropy_hd dsout EntHD mask :1-22@P outfile entropy_vs_length.dat \\\n");
  mprintf("                        details entropy_info.txt c 1.0 kb 1.0 window 500\n\n");
}

Analysis::RetType Analysis_EntropyHD::Setup(ArgList& al, AnalysisSetup& setup, int debugFlag)
{

  mprintf("Analysis_EntropyHD constructor called: %i\n", constructor_called);
  fflush(stdout);

  // ---- Required: output dataset and coordinate dataset names
  dsoutName_  = al.GetStringKey("dsout");
  if (dsoutName_.empty()) {
    mprinterr("entropy_hd: You must specify 'dsout <name>'.\n");
    return Analysis::ERR;
  }
  
  //find the coordinate dataset. FindCoordsSet() can accept an empty string,
  // in which case it will return the first COORDS dataset in the AnalysisSetup::DSL() list.
  coordsName_ = al.GetStringKey("crdset");
  Coords_ = (DataSet_Coords*) setup.DSL().FindCoordsSet(coordsName_);  
  if (Coords_ == nullptr) {
    mprinterr("entropy_hd: Coordinate set '%s' not found.\n", coordsName_.c_str());
    return Analysis::ERR;
  } 

  // ---- Optional args
  maskString_              = al.GetStringKey("mask");
  std::string outfilename  = al.GetStringKey("outfile");

  temp_        = al.getKeyDouble("temp", 300.0);
  pressure_    = al.getKeyDouble("pressure", 1.0);
  window_      = al.getKeyInt("window", 500);
  do_jacobi_   = al.hasKey("jacobi");
  if (window_ <= 0) window_ = 500;

  //this should (neatly and silently) default to stdout.
  {
    std::string outfilename  = al.GetStringKey("outfile");
    outfile_  = setup.DFL().AddCpptrajFile(outfilename, "Entropy Estimate",
                                 DataFileList::TEXT, true);
  }

  // ---- Create output dataset via AnalysisSetup::DSL()
  //
  DataSet* raw = setup.DSL().AddSet(DataSet::DOUBLE, MetaData(dsoutName_.c_str(), -1));
  outputDS_    = dynamic_cast<DataSet_1D*>(raw);
  if (outputDS_ == nullptr) {
    mprinterr("entropy_hd: DataSet type is not a DataSet_1D — check AddSet type.\n");
    return Analysis::ERR;
  }

  mprintf("Analysis_EntropyHD returning OK\n");

  return Analysis::OK;
}

// ---------------------------- low-level math ---------------------------------
void Analysis_EntropyHD::ComputeMean(double* mean, const double* X, int n, int p) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int j = 0; j < p; ++j) {
    double s = 0.0;
    for (int i = 0; i < n; ++i)
      s += X[i * p + j];
    mean[j] = s / (double)n;
  }
}



// ============================================================================
// DEBUG VERSION: Full symmetric eigendecomposition via Jacobi rotations.
// Replaces LogDetChol() temporarily to (1) print eigenvalues and (2) return
// log(det(M)) = sum_i log(lambda_i).
//
// - Input M is assumed symmetric (n x n). This routine DOES NOT overwrite M.
// - Eigenvalues are computed into a local array using Jacobi sweeps.
// - Prints first up to 32 eigenvalues (sorted descending) to mprintf,
//   and (optionally) writes all to a file if ENV ENTROPYHD_EIGFILE is set.
// - Non-positive eigenvalues are guarded: logs a warning and returns a large
//   negative sentinel to make the issue visible upstream.
//
// NOTE: This is for debugging or to verify stability; Jacobi is O(n^3)
// with a large constant.
//
// Matrix diagonalisation is also available elsewhere in the API 
// via "metrix thermo" command.
// ============================================================================
double Analysis_EntropyHD::LogDetJac(double* M, size_t max_evs ) const
{
    // ---- Make a private copy so the caller’s matrix is preserved ----
    std::vector<double> A((size_t)P_ * P_);
    std::memcpy(A.data(), M, (size_t)P_ * P_ * sizeof(double));

    // ---- Jacobi parameters ----
    const int maxIter = 100 * P_;         // generous upper bound
    const double eps  = 1e-14;

    // Diagonal (eigenvalues), work arrays
    std::vector<double> d(P_), b(P_), z(P_);

    // Initialize eigenvalue guesses from the diagonal
    for (size_t i = 0; i < P_; ++i) {
        d[i] = A[(size_t)i * P_ + i];
        b[i] = d[i];
        z[i] = 0.0;
    }

    // Jacobi sweeps
    for (int iter = 0; iter < maxIter; ++iter) {
        // Sum of absolute off-diagonal elements
        double sm = 0.0;
        for (size_t p = 0; p < P_-1; ++p)
            for (size_t q = p+1; q < P_; ++q)
                sm += std::fabs(A[(size_t)p * P_ + q]);

        // Converged?
        if (sm < eps) break;

        const double tresh = (iter < 3 ? 0.2 * sm / (P_ * P_) : 0.0);

        for (size_t p = 0; p < P_-1; ++p) {
            for (size_t q = p+1; q < P_; ++q) {
                const double apq = A[(size_t)p * P_ + q];
                const double g = 100.0 * std::fabs(apq);

                if (iter > 3 &&
                    (std::fabs(d[p]) + g) == std::fabs(d[p]) &&
                    (std::fabs(d[q]) + g) == std::fabs(d[q])) {
                    A[(size_t)p * P_ + q] = 0.0;
                } else if (std::fabs(apq) > tresh) {
                    const double h = d[q] - d[p];
                    double t;
                    if ((std::fabs(h) + g) == std::fabs(h)) {
                        t = apq / h;
                    } else {
                        const double theta = 0.5 * h / apq;
                        t = 1.0 / (std::fabs(theta) + std::sqrt(1.0 + theta * theta));
                        if (theta < 0.0) t = -t;
                    }
                    const double c = 1.0 / std::sqrt(1.0 + t * t);
                    const double s = t * c;
                    const double tau = s / (1.0 + c);
                    const double h2 = t * apq;

                    z[p] -= h2;
                    z[q] += h2;
                    d[p] -= h2;
                    d[q] += h2;
                    A[(size_t)p * P_ + q] = 0.0;

                    // Rotate rows/cols
                    for (size_t j = 0; j < p; ++j) {
                        const double Apj = A[(size_t)j * P_ + p];
                        const double Ajq = A[(size_t)j * P_ + q];
                        A[(size_t)j * P_ + p] = Apj - s * (Ajq + Apj * tau);
                        A[(size_t)j * P_ + q] = Ajq + s * (Apj - Ajq * tau);
                    }
                    for (size_t j = p+1; j < q; ++j) {
                        const double Apj = A[(size_t)p * P_ + j];
                        const double Ajq = A[(size_t)j * P_ + q];
                        A[(size_t)p * P_ + j] = Apj - s * (Ajq + Apj * tau);
                        A[(size_t)j * P_ + q] = Ajq + s * (Apj - Ajq * tau);
                    }
                    for (size_t j = q+1; j < P_; ++j) {
                        const double Apj = A[(size_t)p * P_ + j];
                        const double Aqj = A[(size_t)q * P_ + j];
                        A[(size_t)p * P_ + j] = Apj - s * (Aqj + Apj * tau);
                        A[(size_t)q * P_ + j] = Aqj + s * (Apj - Aqj * tau);
                    }
                }
            }
        }

        // Update diagonal
        for (size_t i = 0; i < P_; ++i) {
            b[i] += z[i];
            d[i]  = b[i];
            z[i]  = 0.0;
        }
    }

    // Sort eigenvalues descending for readability
    std::vector<double> evals = d;
    std::sort(evals.begin(), evals.end(), std::greater<double>());

    // Print a preview
    mprintf("Jacobi: top eigenvalues of M (n=%d):\n", P_);
    const int preview = std::min(int(P_), 32);
    for (int i = 0; i < preview; ++i)
        mprintf("  %3d : %.15e\n", i, evals[i]);
    if (P_ > (size_t)preview) mprintf("  ... (%d total eigenvalues)\n", P_);

    // Compute logdet ignoring non-positive eigenvalues
    double logdet = 0.0;
    for (size_t i = 0; i < P_; ++i) {

      if( i >= max_evs && max_evs > 0 ){
         mprintf("treating matrix as underdetermined, and summing only %lu evs\n", max_evs);
         break;
      }
// Convert Jacobi eigenvalue µ_i of A = I + α C  →  covariance eigenvalue λ_i(C)
// pczdump’s per-mode Schlitter term:  300 * log(1 + 46.03 * λ_i) * unitsFactor
//const double perEv_pcz = temp_ * std::log(1.0 + 46.03 * (temp_ / 300.) * evals[i]) * unitsFactor;

      if (evals[i] > 0.0) {
        logdet += std::log(1.0 + 46.03 * (temp_/300.) * evals[i]); // log(1 + α λ_i) with α=46.03@300K 
      }
    }

    return logdet;
}


// -----------------------------------------------------------------------------
// Eigensolve-free Schlitter log-det 
//
// Computes: ld = log det( I + c_local * C )
// with C the (unbiased) covariance of X (n frames × p dims, row-major).
// By Sylvester: det(I_p + c C) == det(I_n + (c/(n-1)) Xc Xc^T),
// where Xc is X centered along columns.
//
// Complexity: O(n^2 p) to build the Gram, then O(n^3) for Cholesky.
// This is a win whenever n << p (typical MD).
// -----------------------------------------------------------------------------
double Analysis_EntropyHD::SchlitterEntropy(const double* X, size_t n, double *masses ) const
{
  // Prevent crashes on crappy inputs 
  if (X == nullptr || n <= 1 || P_ <= 0 ) return 0.0;

  // column means (length p) 
  double* mean = dalloc(P_);
  if (!mean) return 0.0;

  // mean[j] = average over frames of X[i*p + j]
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t j = 0; j < P_; ++j) mean[j] = 0.0;
  for (size_t i = 0; i < (size_t)n; ++i) {
    const double* row = X + i * P_;
    for (size_t j = 0; j < P_; ++j) mean[j] += row[j];
  }
  const double invN = 1.0 / (double)n;
  for (size_t j = 0; j < P_; ++j) mean[j] *= invN;

  // build A = I_n + alpha * Xc * Xc^T  (n x n SPD) 
  // alpha = c_local / (n-1)  

  mprintf("Building covariance\n");
  fflush(stdout);
  //try working with a normal covariance matrix
  double* C = dalloc((size_t)P_ * P_);
  if ( !C ) {
    mprinterr("Failed to allocate covariance matrix size %lu doubles\n", (size_t)P_ * P_);
    return Analysis::ERR;
  }
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (size_t i = 0; i < P_; ++i) {
    for (size_t j = 0; j < P_; ++j) {
      C[i * P_ + j] = 0.0;
    } 
  }
 
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for (size_t i = 0; i < (size_t)n; ++i){ //loop over frames
    const double* row = X + (size_t)i * P_;
    for (size_t j = 0; j < P_; ++j) { //loop over DOF
      const double xij = row[j] - mean[j];

      for (size_t k = 0; k < P_; ++k) { //loop over DOF
        const double xik = row[k] - mean[k];
        C[j * P_ + k] += xij * xik;
      }
    }
  }

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < P_; ++i)
    for (size_t j = i; j < P_; ++j) {
      double v = C[i*P_ + j] * masses[i] * masses[j] / (n - 1.); // covariance with mass weighting
      C[i*P_ + j] = v;
      C[j*P_ + i] = v;
    } 

  // --- log-det via Cholesky ---
  // If numerical issues, add a tiny ridge to the diagonal and retry.
  //double ld = LogDetChol(A, n);

  mprintf("Getting log determinant of I + Cov by Jacobi\n");
  //this is horribly slow but is better and more stable for number of frames >> number of DOF. 
  double *M = dalloc( P_ * P_ );
  std::memcpy(M, C, sizeof(double) * (size_t)P_ * P_);
  double ldJ = 0.;
  if (  n > P_ - 6 ){
       //if we have more frames than degrees of freedom
       //then the matrix should be fully determined less 6 rigid-body degrees of freedom.
       ldJ = LogDetJac(M, P_ - 6);
  } else {
       //if the matrix is underdetermined, then don't consider junk nar-zero eigenvalues.
       ldJ = LogDetJac(M, n);
  }
  free( M );
  mprintf("                          via Jacobi = %.15e\n", ldJ);
  fflush(stdout);

  std::free(C);
  std::free(mean);

  return ldJ;
}


// Analyze
Analysis::RetType Analysis_EntropyHD::Analyze() {


  // 
  // Validate coordinate data
  // 
  if (Coords_ == nullptr) {
    mprinterr("entropy_hd: No coordinate DataSet loaded.\n");
    return Analysis::ERR;
  }

  const Topology& top = Coords_->Top();  // Get topology reference from coordinates dataset
  const int N         = Coords_->Size(); // number of frames

  mprintf("entropy_hd: running on coordinate set '%s' with %zu frames and %u dimensions.\n",
            coordsName_.c_str(), Coords_->Size(), Coords_->Ndim());
  if (N < window_) {
    mprinterr("entropy_hd: Not enough frames (%d) for window=%d.\n", N, window_);
    return Analysis::ERR;
  }


  // ---------------------------------------
  // Setup atom mask
  // ---------------------------------------
  if (!maskString_.empty()) {
    if (mask_.SetMaskString(maskString_) != 0) {
        mprinterr("entropy_hd: Invalid mask '%s'.\n", maskString_.c_str());
        return Analysis::ERR;
    }
  } else {
    // default = all atoms
    mask_.SetMaskString("@*");
  }
  
  // Now apply mask to topology
  if (top.SetupIntegerMask(mask_) != 0) {
    mprinterr("entropy_hd: Failed to apply mask to topology.\n");
    return Analysis::ERR;
  }
  mask_.MaskInfo();
  if( mask_.None() ) {
    mprinterr("Error: mask [%s] is somehow None.\n", mask_.MaskString());
    return Analysis::ERR;
  }
  nSelected_ = mask_.Nselected();
  if (nSelected_ <= 0) {
    mprinterr("entropy_hd: Mask selects zero atoms.\n");
    return Analysis::ERR;
  }
  
  //get a mass vector for mass-weighting covariance later.
  P_ = 3 * nSelected_;

  // ------------------------------------------------------------
  // Build per-coordinate mass vector (length p = 3*nSelected_)
  // ------------------------------------------------------------
  std::vector<double> massVec(P_);   // P_ = 3*nSelected_
  std::vector<Atom>   atsInMask;
  atsInMask.reserve( nSelected_ );
  for (int si = 0; si < nSelected_; si++) {
    int atom = mask_[si];
    double m = top[atom].Mass();          // mass in amu

    // Basic sanity check for hydrogens
    if (top[atom].AtomicNumber() == 1 && m > 2.01) {
        mprinterr("entropy_hd: Hydrogen atom mass %.2f > 2.01; suspect topology?\n");
        mprinterr("    If you used HMR then you have to reverse it before calculating an entropy. Quitting.\n", m);
        return Analysis::ERR;
    }
    // Assume that each atom contributes three coordinate entries
    massVec[3*si + 0]   = std::sqrt(m);
    massVec[3*si + 1]   = std::sqrt(m);
    massVec[3*si + 2]   = std::sqrt(m);

    atsInMask.push_back( top[atom] );
  }
 


  // ---------------------------------------
  // Stage selected coordinates into memory,
  // align and centre
  // ---------------------------------------
  mprintf("allocating memory for coordinate matrix X with %d rows and %d columns...\n", N, P_);
  double* X = dalloc((size_t)N * P_);
  if (X == nullptr) {
    mprinterr("entropy_hd: Failed to allocate memory for coordinate matrix.\n");
    return Analysis::ERR;
  } else {
    mprintf("dalloc'd. Aligning coordinates:\n");
  }


  //this saves the average coordinates to AverageFrame_
  BuildAlignedCoordinates(Coords_, mask_, X);
  mprintf("Aligned.\n");

  //for convenience, the AverageFrame has unit mass atoms
  //because all the averaging is geometric, but we set them
  //to have real masses for the rigid body stuff later.
  AverageFrame_.SetMass( atsInMask );
  //after this point, shouldn't need to reference anything to do with the mask: have all masses and coords.

  // ---------------------------------------
  // Compute Schlitter constant c(T) in AMBER units
  // ---------------------------------------
  static constexpr double kB_kcal     = 0.00198720425864083;
  static constexpr double hbar_kcalps = 0.0635078;

  // ---------------------------------------
  // Compute number of windows
  // ---------------------------------------
  size_t K = N / window_;
 
  mprintf("using %d windows of size %d\n", K, window_);
  if ( K < 3 ) {
    mprinterr("entropy_hd: Not enough frames (%d) for window=%d to produce at least 3 windows; need at least three.\n", N, window_);
    std::free(X);
    return Analysis::ERR;
  }

  std::vector<double> nvals(K), Svals(K);

  // ---------------------------------------
  // Compute S(n) for each window over the trajectory 
  //
  // TODO: make a circular buffer instead of always
  // counting windows from the start of the trajectory?
  // ---------------------------------------
  const double entropy_units = 0.5 * (8.314 / 4200.0);
  for (size_t i = 0; i < K; ++i) {
    size_t n = (i + 1) * window_;
    double ld;
    ld = Stable_Schlitter_LogDet( X, n, P_, massVec.data() );
    if ( do_jacobi_ ) {
      mprintf(" log determinant by Cholesky: %f\n", ld);
      ld = SchlitterEntropy( X, n, massVec.data() );
      mprintf("Jacobi log determinant: %f\n . Saving Jacobi as it was requested.", ld);
    }

    Svals[i] = entropy_units * temp_ * ld; // free-energy units (kcal/mol);
    nvals[i] = (double)n;

    mprintf("window %d: n=%d frames, S(n)=%.4f kcal/mol @ T=%.1f K\n", i + 1, n, Svals[i], temp_);
    fflush( stdout );

  }

  // write K time points to DataSet
  outputDS_->SetDim(Dimension::X, Dimension(nvals[0], window_, "X"));
  for (size_t i = 0; i < K; ++i) {
    outputDS_->Add(i, &Svals[i]);
  }

  // build thermo-style output of rotational and translational energy and entropy
  // and save out detailed report
  outfile_->Printf( "#Harris-Dryden unbiased entropy estimator given time series of Molecular Dynamics Frames:\n");
  outfile_->Printf( "#References:\n");
  outfile_->Printf( "#   - Dryden, Kume, Le, Wood (Stat. inference for functions of covariance...,\n");
  outfile_->Printf( "#     Sections 4.1–4.2, Schlitter entropy and bias behavior).  \n");
  outfile_->Printf( "#   - Harris et al. (2001) empirical bias scaling for S(nr S(n). \n");
  outfile_->Printf( "#Mask: %s\n", maskString_.empty() ? "<all atoms>" : maskString_.c_str());
  outfile_->Printf( "#Selected atoms: %i (P=%lu)\n", nSelected_, P_);

  ////////Header printout and rigid body stuff (common to entropy calculators) is taken care of by ThermoUtils
  //pack state into this ThermoInput structure, and pass it as an argument
  ThermoUtils::ThermoInput tin;
  tin.masses_amu       = &MassesPerAtom_; //pass by reference of vector.
  tin.avg_coords       = &AvgXYZ_;        //pass by reference of vector.
  tin.freqs_cm1        = nullptr;   //this method doesn't generate a spectrum
  tin.nmodes           = 0;         //no modes
  tin.temperature      = temp_;     //scalar
  tin.pressure_atm     = pressure_; //scalar

  //print out the thermochemistry
  ThermoUtils::ComputeThermochemistry( *outfile_, tin, 0 );


  outfile_->Printf("\n#####Configurational entropy estimate versus time (better sampling makes the entropy appear larger)\n");
  outfile_->Printf(  "#below timeseries converges (probably) as S(t) = S_inf( 1 - exp(-B * t))...\n");
  outfile_->Printf(  "#  : the user is advised to estimate the value S_inf using 'regress YourDataSetName model expsat'\n");
  outfile_->Printf(  "#n     S(n)        \n");
  for (size_t i = 0; i < K; ++i)
     outfile_->Printf( "%lu %.6f\n", (long unsigned int)nvals[i], Svals[i]);

  std::free(X);
  return Analysis::OK;
}


void Analysis_EntropyHD::BuildAlignedCoordinates(
        DataSet_Coords* Coords_,
        const AtomMask& mask_,
              double* X )      // OUTPUT: (nFrames × p), RB-removed, centered
{
    const int nFrames = Coords_->Size();
    const int nSel    = mask_.Nselected();
    const int p       = 3 * nSel;
    Frame     ref, fr;

    ref.SetupFrameFromMask(mask_);
    fr.SetupFrameFromMask(mask_);

    // -----------------------------------------------------------
    // First pass: align to frame 0 and accumulate Cartesian average
    // -----------------------------------------------------------
    Coords_->GetFrame(0, ref, mask_);

try {
    AvgXYZ_.resize(p, 0.); 
} catch (const std::bad_alloc& e)  {
    mprinterr("couldn't allocate a vector of %i doubles \n", p );
    exit( 1 );
}
    for (int f = 0; f < nFrames; f++) {
        Coords_->GetFrame(f, fr, mask_);

        fr.Align(ref, mask_);   

        for (int si = 0; si < nSel; si++) {
            const double* xyz = fr.XYZ(si);
            AvgXYZ_[3*si + 0] += xyz[0];
            AvgXYZ_[3*si + 1] += xyz[1];
            AvgXYZ_[3*si + 2] += xyz[2];
        }
    }


    const double invN = 1.0 / double(nFrames);
    for (int j = 0; j < p; j++) AvgXYZ_[j] *= invN;

    // -----------------------------------------------------------
    // Build average frame as a "Frame" object from first-pass mean
    // NB it is not clear if this is a copy-by-ref: better keep the mass vector
    // and the AvgXYZ_ in scope for the rest of the analysis.
    // -----------------------------------------------------------

    MassesPerAtom_.resize(nSel, 1.0);
    AverageFrame_.SetupFrameXM(AvgXYZ_, MassesPerAtom_); //use unit mass initially, for geometric averaging only.

    // -----------------------------------------------------------
    // Second pass: align to avgFr, then:
    //   - remove translation (COM),
    //   - remove rotation via projection (3 rotational modes),
    //   - store aligned frames in X.
    //   - also store average coordinates (for thermo).
    // -----------------------------------------------------------
#pragma omp parallel
    {
        Frame local;
        local.SetupFrameFromMask(mask_);

        // Per-thread temporaries
        std::vector<double> y(p);        // centered coordinates (to be rotation-free)
        std::vector<double> Bx(p), By(p), Bz(p);  // rotational bases 
        double G[9];     // Gram = [Bx By Bz]^T * [Bx By Bz], 3x3
        double rhs[3];   // B^T y
        double c[3];     // coefficients

#pragma omp for schedule(static)
        for (int f = 0; f < nFrames; f++) {
	
//marking the GetFrame as critical because I'm not sure if it mutates state inside the coords reader.
#pragma omp critical
		{
            Coords_->GetFrame(f, local, mask_);
		}

            local.Align(AverageFrame_, mask_); //because we copied the average coordinates into the Frame object, 
	                               //we can use API functions like Align()

            // ---------- Translation removal:  COM ----------
            // COM = sum(m_i * x_i) / sum(m_i)
            double comX = 0.0, comY = 0.0, comZ = 0.0;
            for (int si = 0; si < nSel; si++) {
                const double* xyz = local.XYZ(si);
                comX +=  xyz[0];
                comY +=  xyz[1];
                comZ +=  xyz[2];
            }
            comX /= nSel; comY /= nSel; comZ /= nSel;
            

            // ---------- Build centered y and rotational bases Bx,By,Bz ----------
            // r_i = x_i - COM; y(3si:3si+2) = sqrt(m_i)*r_i
            // b_x = sqrt(m_i) * ( e_x × r ) = sqrt(m_i) * ( 0, -r_z, r_y )
            // b_y = sqrt(m_i) * ( e_y × r ) = sqrt(m_i) * ( r_z, 0, -r_x )
            // b_z = sqrt(m_i) * ( e_z × r ) = sqrt(m_i) * ( -r_y, r_x, 0 )
            for (int si = 0; si < nSel; si++) {
                const double* xyz = local.XYZ(si);
                const double rx = xyz[0] - comX;
                const double ry = xyz[1] - comY;
                const double rz = xyz[2] - comZ;
                const int jx = 3*si;

                // centered coordinates
                y[jx+0] =  rx;
                y[jx+1] =  ry;
                y[jx+2] =  rz;

                // rotational bases 
                Bx[jx+0] =  0.0;
                Bx[jx+1] = -rz;
                Bx[jx+2] =  ry;

                By[jx+0] =  rz;
                By[jx+1] =  0.0;
                By[jx+2] = -rx;

                Bz[jx+0] = -ry;
                Bz[jx+1] =  rx;
                Bz[jx+2] =  0.0;
            }

            // ---------- Build Gram G = B^T B and rhs = B^T y (3x3 and 3x1) ----------
            // G = [ <Bx,Bx>, <Bx,By>, <Bx,Bz>;
            //       <By,Bx>, <By,By>, <By,Bz>;
            //       <Bz,Bx>, <Bz,By>, <Bz,Bz> ]
            double Gxx=0, Gxy=0, Gxz=0, Gyy=0, Gyz=0, Gzz=0;
            double rBx=0, rBy=0, rBz=0;

            for (int j = 0; j < p; j++) {
                const double bx = Bx[j], by = By[j], bz = Bz[j], yy = y[j];
                Gxx += bx*bx; Gxy += bx*by; Gxz += bx*bz;
                Gyy += by*by; Gyz += by*bz; Gzz += bz*bz;
                rBx += bx*yy; rBy += by*yy; rBz += bz*yy;
            }
            G[0]=Gxx; G[1]=Gxy; G[2]=Gxz;
            G[3]=Gxy; G[4]=Gyy; G[5]=Gyz;
            G[6]=Gxz; G[7]=Gyz; G[8]=Gzz;

            rhs[0]=rBx; rhs[1]=rBy; rhs[2]=rBz;

            // ----------  Solve G c = rhs (3x3) with tiny Tikhonov regularization ----------
            // Add a small ridge in case the geometry is near-degenerate (e.g., nearly linear subset)
            const double eps = 1e-12;
            G[0]+=eps; G[4]+=eps; G[8]+=eps;

            // Invert 3x3 via adjugate
            const double A=G[0], B=G[1], C=G[2];
            const double D=G[3], E=G[4], F=G[5];
            const double H=G[7], I=G[8]; // (G[6]=Gxz reused via C)
            const double det = A*(E*I - F*H) - B*(D*I - F*C) + C*(D*H - E*C);
            double invG[9] = {0};
            if (std::fabs(det) > 1e-30) {
                invG[0] =  (E*I - F*H) / det;
                invG[1] = -(B*I - C*H) / det;
                invG[2] =  (B*F - C*E) / det;
                invG[3] = -(D*I - F*C) / det;
                invG[4] =  (A*I - C*C) / det;
                invG[5] = -(A*F - C*D) / det;
                invG[6] =  (D*H - E*C) / det;
                invG[7] = -(A*H - B*C) / det;
                invG[8] =  (A*E - B*D) / det;

                c[0] = invG[0]*rhs[0] + invG[1]*rhs[1] + invG[2]*rhs[2];
                c[1] = invG[3]*rhs[0] + invG[4]*rhs[1] + invG[5]*rhs[2];
                c[2] = invG[6]*rhs[0] + invG[7]*rhs[1] + invG[8]*rhs[2];
            } else {
                // Near singular: fall back to no rotational correction
                c[0]=c[1]=c[2]=0.0;
            }

            // ---------- Project rotation: y <- y - (c0*Bx + c1*By + c2*Bz) ----------
            for (int j = 0; j < p; j++)
                y[j] = y[j] - (c[0]*Bx[j] + c[1]*By[j] + c[2]*Bz[j]);

            // ---------- Store RB-removed coordinates ----------
            double* row = X + (size_t)f * p;
            for (int j = 0; j < p; j++) row[j] = y[j];
        }
    }

    // -----------------------------------------------------------
    // SECOND-PASS MEAN and CENTERING
    // -----------------------------------------------------------
    // could probably, if pushed, re-use the space AvgXYZ_ which was previously
    // calculated
    std::vector<double> Avg2XYZ( p );
    for (int j = 0; j < p; j++) Avg2XYZ[j] = 0.0;

    int nThreads = 1;
#pragma omp parallel
    {
#pragma omp single
      nThreads = omp_get_num_threads();
    }

    std::vector<std::vector<double>> threadSum(nThreads, std::vector<double>(p, 0.0));

#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      double* local = threadSum[tid].data();

#pragma omp for schedule(static)
      for (int f = 0; f < nFrames; f++) {
         const double* row = X + (size_t)f * p;
         for (int j = 0; j < p; j++)
           local[j] += row[j];
      }
    }
    for (int t = 0; t < nThreads; t++)
      for (int j = 0; j < p; j++)
        Avg2XYZ[j] += threadSum[t][j];

    //reduce over threads to get the Cartesian (not mw)
    //centred and aligned coordinates.
    for (int j = 0; j < p; j++)
      Avg2XYZ[j] *= invN;

#pragma omp parallel for schedule(static)
    for (int f = 0; f < nFrames; f++) {
      double* row = X + (size_t)f * p;
      for (int j = 0; j < p; j++)
        row[j] -= Avg2XYZ[j];
    }

    //save the average frame. Maybe even export this as an output?
    //NB this might well not be neccessary as the AvgXYZ_ buffer was already linked into 
    //the AverageFrame_ object but it isn't expensive so do it anyway. 
    for( int i = 0; i < nSel; i++ ){
      AverageFrame_.SetXYZ( i, Vec3( Avg2XYZ[3*i], Avg2XYZ[3*i+1], Avg2XYZ[3*i+2] ) );
    }
}


// --------------------------------------------------------------
// Helper: lightweight matrix accessor (const + mutable)
// --------------------------------------------------------------
struct MatrixView {
    double* data;
    size_t n;
    inline double& operator()(size_t r, size_t c) {
        return data[r*n + c];
    }
    inline const double& operator()(size_t r, size_t c) const {
        return data[r*n + c];
    }
};

// Gershgorin + diag + symmetry diagnostics
static void diag_report(const MatrixView& A, const char* tag) {
    size_t n = A.n;
    size_t nonpos_diag = 0;
    double min_diag = +1e300, max_diag = -1e300;
    double max_asym = 0.0, max_off = 0.0;
    double gersh_min = +1e300;

    for (size_t i = 0; i < n; ++i) {
        double di = A(i,i);
        if (!(di > 0.0)) nonpos_diag++;
        min_diag = std::min(min_diag, di);
        max_diag = std::max(max_diag, di);

        double radius = 0.0;
        for (size_t j = 0; j < n; ++j) {
            double aij = A(i,j);
            double aji = A(j,i);
            max_asym = std::max(max_asym, std::fabs(aij - aji));
            if (i != j) {
                double off = std::fabs(aij);
                max_off = std::max(max_off, off);
                radius += off;
            }
        }
        gersh_min = std::min(gersh_min, di - radius);
    }

//technical estimates of matrix stability
//    mprintf(">>> DIAG(%s): n=%zu  nonpos_diag=%zu  min_diag=% .6e  max_diag=% .6e\n",
//            tag, n, nonpos_diag, min_diag, max_diag);
//    mprintf("              max|A-A^T|=% .6e  max|off|=% .6e  Gershgorin_min=% .6e\n",
//            max_asym, max_off, gersh_min);

}


// --------------------------------------------------------------
// Kahan‑compensated greedy pivoted Cholesky + log det
// Works for both Gram and Cov matrices
// --------------------------------------------------------------
static double greedy_pivoted_cholesky_logdet(
        MatrixView& A,
        bool verbose = false,
        double eps_pivot = 1e-12)
{
    const size_t n = A.n;

    // Permutation vector
    std::vector<size_t> piv(n);
    for (size_t i = 0; i < n; ++i) piv[i] = i;

    // Triangular compensation buffer
    std::vector<double> comp(n*(n+1)/2, 0.0);
    auto idxT = [&](size_t i, size_t j) -> size_t {
        if (i < j) std::swap(i,j);
        return (i*(i+1))/2 + j;
    };

    std::vector<double> row(n);
    double logd = 0.0;

    for (size_t k = 0; k < n; ++k) {

        // ---- pivot search ----
        double maxDiag = -1e300;
        size_t p = k;
        for (size_t i = k; i < n; ++i) {
            double d = A(piv[i], piv[i]);
            if (d > maxDiag) { maxDiag = d; p = i; }
        }

        if (maxDiag < eps_pivot) {
            if (verbose) {
                mprintf("PivotedChol: stop at k=%zu, maxDiag=% .6e\n", k, maxDiag);
            }
            return 2.0 * logd;
        }

        // ---- apply pivot ----
        std::swap(piv[k], piv[p]);
        const size_t pk = piv[k];

        // ---- extract pivot ----
        const double Akk = A(pk, pk);
        if (!(Akk > 0.0)) {
            if (verbose)
                mprintf("PivotedChol: non-positive pivot at k=%zu: % .6e\n", k, Akk);
            return 2.0 * logd;
        }
        const double Lkk = std::sqrt(Akk);
        A(pk, pk) = Lkk;
        logd += std::log(Lkk);

        const double invLkk = 1.0 / Lkk;

        // ---- build row ----
        for (size_t j = k+1; j < n; ++j) {
            size_t pj = piv[j];
            row[j] = A(pk, pj) * invLkk;
        }

        // ---- Kahan‑compensated Schur update ----
        for (size_t i = k+1; i < n; ++i) {
            size_t pi = piv[i];
            double ri = row[i];

            for (size_t j = i; j < n; ++j) {
                size_t pj = piv[j];
                double rj = row[j];
                double term = -(ri * rj);

                double& aij = A(pi, pj);
                double& cij = comp[idxT(pi,pj)];

                double delta = term - cij;
                double t = aij + delta;
                cij = (t - aij) - delta;
                aij = t;

                A(pj, pi) = aij; // keep symmetry
            }
        }
    }
    if (verbose) {
       mprintf("PivotedChol: completed at k=n=%zu, log det=% .6e\n", n, 2*logd);
    }

    return 2.0 * logd;
}


// --------------------------------------------------------------
// Main exported function: automatic Gram/full switching
// --------------------------------------------------------------
double Analysis_EntropyHD::Stable_Schlitter_LogDet(
        const double* Xc,   // centered & RB‑removed coordinates, (n×p)
        const size_t  n,                // frames
        const size_t  p,                //DOF 
	const double *masses )   //sqrt mass per dof
{
    const double alpha = 46.03 * (temp_/300.0);
    double      *xcmw, *xmean;  
    size_t       ptr_save;

    xcmw   = dalloc( p * n );
    xmean  = dalloc( p );
    if (xcmw == NULL || xmean == NULL ){
      mprinterr("failed to allocate memory for mass-weighted trajectory\n");
      return 0.;
    }


    //need to re-centre as the centroid of the sub-window will
    //different to that for the whole traj
    for ( size_t i = 0; i < p ; i++ ){
      xmean[i] = 0.;
    }

    //collect frames from a circular buffer up to the number
    //specified. Maximising independence of data.
    ptr_save = read_ptr_;
    for ( size_t f = 0; f < n; f++ ){
      for ( size_t i = 0; i < p; i++ ){
        xmean[i] += Xc[read_ptr_*p+i];
      }
      read_ptr_ += 1;
      if ( read_ptr_ >= n ) read_ptr_ = 0;
    }
    read_ptr_ = ptr_save; //reset the circular buffer.

    for ( size_t i = 0; i < p ; i++ ){
      xmean[i] /= n;
    }

    //repeat the pass over the circular buffer and collect centred frames.
    for ( size_t f = 0; f < n; f++ ){
      for ( size_t i = 0; i < p; i++ ){
        xcmw[f*p+i] = ( Xc[read_ptr_*p+i] - xmean[i] ) * masses[i];
      }
      read_ptr_ += 1;
      if ( read_ptr_ >= n ) read_ptr_ = 0;
    }
    free( xmean );
    //do not reset the buffer again.

    // --------------------------------------------
    // Auto‑switch: Gram if n < p-6, Cov if n >= p‑6
    // --------------------------------------------
    bool useGram = (n < (p - 6));

    if (useGram) {

        // Form G = I + α/(n-1) * Xc Xc^T
        std::vector<double> Gv(n*n, 0.0);
        MatrixView G{Gv.data(), n};

        const double scale = alpha / double(n - 1);

	#pragma omp parallel for schedule(static)
	//outer loop is over frames i
        for (size_t i = 0; i < n; ++i) {
	    size_t ii;
	    ii = ((i + ptr_save) % n) * p;

	    //copy the outer-loop frame into cache, 
	    //hope that it fits per-thread.
            std::vector<double> xi(p);
            for (size_t k = 0; k < p; ++k) xi[k] = xcmw[ii+k];

	    //inner loop over frames j, should synchronise such that all
	    //threads load the same inner set of frames to cache.
	    for (size_t j = 0; j < n; ++j) {
              size_t jj  = ((j + ptr_save) % n) * p;
              double sum = 0.;

              //now loop over degrees of freedom for the frame-pair.
              for (size_t k = 0; k < p; ++k) {
                double xjk  =  xcmw[jj + k];
                sum        +=  xi[k] * xjk;
              }
	      G(i,j) = sum * scale;
           }
	   //add identity
	   G(i,i) += 1.0;
       }

//        diag_report(G, "GRAM pre-factor");

        free( xcmw );

        return greedy_pivoted_cholesky_logdet( G );
    }

    // ---------------------------------------------------------
    // FULL COV MATRIX path: p×p
    // ---------------------------------------------------------
 //   mprintf("Stable Schlitter: guessing we have enough frames for a stable covariance matrix: (%zu×%zu)\n", p, p);

    std::vector<double> Cv(p*p, 0.0);
    MatrixView C{Cv.data(), p};
    
    // Build covariance: C(i,j) = sum_f row_f[i] * row_f[j]
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < p; ++i) {
      for (size_t j = i; j < p; ++j) {
        double sum = 0.0;
        // accumulate over frames: all threads should load new frames in synchrony,
	// so although it means a lot (N2) of frame loads at least they aren't fighting each other.
        for (size_t f = 0; f < n; ++f) {
            const double* row = xcmw + f*p;
            sum += row[i] * row[j];
        }
        C(i,j) = sum;
      }
    }

    // symmetrise and scale 
    double scaled_inv = alpha / double(n-1);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < p; ++i) {
      for (size_t j = i; j < p; ++j) {
        double v = C(i,j) * scaled_inv;
        C(i,j) = v;
        C(j,i) = v;
      }
    }

    // add Identity.
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < p; ++i) {
      C(i,i) += 1.0;
    }



  //  diag_report(C, "COV pre-factor");

    free( xcmw );

    return greedy_pivoted_cholesky_logdet( C );
}


















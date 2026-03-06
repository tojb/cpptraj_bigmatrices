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
// Using pczdump’s Schlitter as the physical reference (i.e., a sum of log⁡(1+α λi)\log(1+\alpha\,\lambda_i)log(1+αλi​) over covariance eigenvalues) 
// we here compute the same quantity without any eigensolve by switching to a frame‑space Gram formulation and a small Cholesky log‑det. 
// This is mathematically justified by Sylvester’s determinant identity: 
//  det⁡ ⁣(IP+α XcTXc)  =  det⁡ ⁣(IN+α XcXcT)\det\!\big(I_P + \alpha\,X_c^{\mathsf T}X_c\big) \;=\;
//                      \det\!\big(I_N + \alpha\,X_cX_c^{\mathsf T}\big)det(IP​+αXcT​Xc​)=det(IN​+αXc​XcT​)
// so we can work in N×NN\times NN×N (frames) rather than P×PP\times PP×P (coordinates). 
// This delivers an eigensolve‑free Schlitter that is directly comparable to pczdump’s 
// ∑ilog⁡(1+α λi)\sum_i \log(1+\alpha\,\lambda_i)∑i​log(1+αλi​) because both are exactly log⁡det⁡(I+α C)\log\det(I+\alpha\,C)logdet(I+αC).
//
// This is NOT directly comparable to the output of "diagmatrix thermo" because that uses a different formula for the entropy per degree
// of freedom, based on the Quantum Harmonic Oscillator, which is unfortunately not easily expressed in a Gram formulation and thus does require an eigensolve.
//
// Deltas of entropy between different states or conditions should be comparable across both methods, however, since the different fudge factors 
// employed to avoid taking log of zero or negative eigenvalues should largely cancel out. The "fudge factor" in the QHO formula is a constant
// added to the eigenvalues, with some physical justification, 
// while the "fudge factor" in the Schlitter formula is a constant multiplied by the covariance matrix, 
// with an unclear physical justification if any, so they are not directly comparable, 
// but both serve to regularize the entropy estimate.
//
// The advanced feature here is the Harris–Dryden bias correction, which fits the observed S(n) to the empirical scaling form S(n) = S0 + m * n^{-a} 
// and extrapolates as n→∞ to get S0, which is a less biased estimate of the true configurational entropy.
//
// 05/03/2026 Joshua T Berryman, josh.berryman@uni.lu.
//
//



#include <omp.h>
#include <cstring>              // std::memcpy()
#include "CpptrajStdio.h"      // mprintf(), mprinterr()
#include "DataSet.h"
#include "DataSet_MatrixDbl.h"
#include "DataSetList.h"       // Dsets()
#include "DataFileList.h"
#include "DataSet_Coords.h"
#include "DataSet_MatrixDbl.h"
#include "Topology.h"
#include "Frame.h"
#include "AtomMask.h"
#include "Analysis_EntropyHD.h"


static constexpr double kB   = 0.00198720425864083;   //kcal/mol/K
static constexpr double hbar = 0.0635078          ;   //kcal*ps/mol   

static int constructor_called = 0; // For testing whether constructor is called in unit tests. Remove when no longer needed.

// --------------------------------- constructor --------------------------------------
Analysis_EntropyHD::Analysis_EntropyHD() :
    dsoutName_(""),
    coordsName_(""),
    maskString_(""),
    outfileName_(""),
    detailsName_(""),
    temp_(300.0),
    window_(500),
    Coords_(nullptr),
    outputDS_(nullptr),
    nSelected_(0),
    P_(0)
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
  mprintf("                        [c <val>] [kb <val>] [window <frames>]\n\n");
  mprintf("OPTIONS:\n");
  mprintf("  dsout   <name>     Name of output DataSet.\n");
  mprintf("  mask    <atoms>    AMBER atom mask to select coordinates (default: all atoms).\n");
  mprintf("  outfile <file>     Write ASCII table with columns below (optional).\n");
  mprintf("  details <file>     Write verbose report with parameters and results (optional).\n");
  mprintf("  temp    <val>      Temperature in Kelvin (default 300.0).\n");
  mprintf("  window  <frames>   Window size for n-grid; rows at n = window, 2*window, ... (default 500).\n\n");
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

Analysis::RetType Analysis_EntropyHD::Setup(ArgList& al, AnalysisSetup& setup, int debugIn)
{

  mprintf("Analysis_EntropyHD constructor called: %i\n", constructor_called);

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
  maskString_  = al.GetStringKey("mask");
  outfileName_ = al.GetStringKey("outfile");
  detailsName_ = al.GetStringKey("details");
  temp_        = al.getKeyDouble("temp", 300.0);
  window_      = al.getKeyInt("window", 500);
  if (window_ <= 0) window_ = 500;

  // ---- Create output dataset via AnalysisSetup::DSL()
  DataSet* ds = setup.DSL().AddSet(DataSet::MATRIX_DBL, dsoutName_);
  if (ds == nullptr) {
    mprinterr("entropy_hd: Failed to create dataset '%s'.\n", dsoutName_.c_str());
    return Analysis::ERR;
  }
  outputDS_ = static_cast<DataSet_MatrixDbl*>(ds);

  // (Optional) If you want to attach output files here, follow built-in pattern:
  // DataFile* ascii = setup.DFL().AddDataFile(outfileName_, al);
  // if (ascii) ascii->AddDataSet(outputDS_);
  
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

// ---------------------------- low-level math ---------------------------------
void Analysis_EntropyHD::ComputeCov(double* C,
                                    const double* X,
                                    const double* mean,
                                    int n, int p)
{
  // Zero the covariance matrix
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (int i = 0; i < p; ++i)
    for (int j = 0; j < p; ++j)
      C[i * p + j] = 0.0;

  // Accumulate upper triangle (then mirror)
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < p; ++i) {
    for (int j = i; j < p; ++j) {
      double s = 0.0;
      for (int k = 0; k < n; ++k) {
        const double xi = X[k * p + i] - mean[i];
        const double xj = X[k * p + j] - mean[j];
        s += xi * xj;
      }
      s /= static_cast<double>(n);
      C[i * p + j] = s;
      C[j * p + i] = s;
    }
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
// NOTE: This is for debugging only; Jacobi is O(n^3) with a large constant.
// ============================================================================
double Analysis_EntropyHD::LogDetJac(double* M, int n, const double c_local)
{
    // ---- Make a private copy so the caller’s matrix is preserved ----
    std::vector<double> A((size_t)n * n);
    std::memcpy(A.data(), M, (size_t)n * n * sizeof(double));

    // ---- Jacobi parameters ----
    const int maxIter = 100 * n;         // generous upper bound
    const double eps  = 1e-14;

    // Diagonal (eigenvalues), work arrays
    std::vector<double> d(n), b(n), z(n);

    // Initialize eigenvalue guesses from the diagonal
    for (int i = 0; i < n; ++i) {
        d[i] = A[(size_t)i * n + i];
        b[i] = d[i];
        z[i] = 0.0;
    }

    // Jacobi sweeps
    for (int iter = 0; iter < maxIter; ++iter) {
        // Sum of absolute off-diagonal elements
        double sm = 0.0;
        for (int p = 0; p < n-1; ++p)
            for (int q = p+1; q < n; ++q)
                sm += std::fabs(A[(size_t)p * n + q]);

        // Converged?
        if (sm < eps) break;

        const double tresh = (iter < 3 ? 0.2 * sm / (n * n) : 0.0);

        for (int p = 0; p < n-1; ++p) {
            for (int q = p+1; q < n; ++q) {
                const double apq = A[(size_t)p * n + q];
                const double g = 100.0 * std::fabs(apq);

                if (iter > 3 &&
                    (std::fabs(d[p]) + g) == std::fabs(d[p]) &&
                    (std::fabs(d[q]) + g) == std::fabs(d[q])) {
                    A[(size_t)p * n + q] = 0.0;
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
                    A[(size_t)p * n + q] = 0.0;

                    // Rotate rows/cols
                    for (int j = 0; j < p; ++j) {
                        const double Apj = A[(size_t)j * n + p];
                        const double Ajq = A[(size_t)j * n + q];
                        A[(size_t)j * n + p] = Apj - s * (Ajq + Apj * tau);
                        A[(size_t)j * n + q] = Ajq + s * (Apj - Ajq * tau);
                    }
                    for (int j = p+1; j < q; ++j) {
                        const double Apj = A[(size_t)p * n + j];
                        const double Ajq = A[(size_t)j * n + q];
                        A[(size_t)p * n + j] = Apj - s * (Ajq + Apj * tau);
                        A[(size_t)j * n + q] = Ajq + s * (Apj - Ajq * tau);
                    }
                    for (int j = q+1; j < n; ++j) {
                        const double Apj = A[(size_t)p * n + j];
                        const double Aqj = A[(size_t)q * n + j];
                        A[(size_t)p * n + j] = Apj - s * (Aqj + Apj * tau);
                        A[(size_t)q * n + j] = Aqj + s * (Apj - Aqj * tau);
                    }
                }
            }
        }

        // Update diagonal
        for (int i = 0; i < n; ++i) {
            b[i] += z[i];
            d[i]  = b[i];
            z[i]  = 0.0;
        }
    }

    // Sort eigenvalues descending for readability
    std::vector<double> evals = d;
    std::sort(evals.begin(), evals.end(), std::greater<double>());

    // Print a preview
    mprintf("DEBUG(Jacobi): top eigenvalues of M (n=%d):\n", n);
    const int preview = std::min(n, 32);
    for (int i = 0; i < preview; ++i)
        mprintf("  %3d : %.15e\n", i, evals[i]);
    if (n > preview) mprintf("  ... (%d total eigenvalues)\n", n);

    // Compute logdet and guard non-positive eigenvalues
    double logdet = 0.0;
    int nonpos = 0;
    double running_total = 0.0; // For debugging Schlitter contributions
    float  unitsFactor = ( (0.5*8.314)/(4.2*1000) ); /* convert to kcal/mol */

    const double alpha = c_local / double(n - 1);
    mprintf("DEBUG: alpha(c_local/(n-1)) = %.15e (n=%d)\n", alpha, n);


    // Optional: write all eigenvalues to file if an env var is set
    FILE* F = std::fopen("evdump_debug.txt", "w");

    for (int i = 0; i < n; ++i) {


// Convert Jacobi eigenvalue µ_i of A = I + α C  →  covariance eigenvalue λ_i(C)
const double lambda = (evals[i] - 1.0) / alpha;

// pczdump’s per-mode Schlitter term:  300 * log(1 + 46.03 * λ_i) * unitsFactor
const double perEv_pcz = 300.0 * std::log(1.0 + 46.03 * evals[i]) * unitsFactor;

if ( evals[i] > 0.0 ) {
    // Running total (accumulate like pczdump does)
    running_total += std::log(1.0 + 46.03 * lambda);
    //mprintf("DEBUG: λ_i = %.6e, perEv_pcz = %.6e, running_total = %.6e\n", lambda, perEv_pcz, 300.0 * running_total * unitsFactor);
}

        std::fprintf(F, "%.17e\n", perEv_pcz);

        if (evals[i] <= 0.0) {
            nonpos++;
        } else {
            logdet += std::log(1.0 + 46.03 * evals[i]); // log(1 + α λ_i) with α=46.03 for pczdump compatibility
        }
    }
    std::fclose(F);
    if (nonpos > 0)
        mprinterr("DEBUG(Jacobi): %d non-positive eigenvalue(s) encountered; "
                  "logdet will be very negative.\n", nonpos);

    return logdet;
}

double Analysis_EntropyHD::LogDetChol(double* M, int p)
{
  double logd = 0.0;

  for (int j = 0; j < p; ++j) {
    // Diagonal element update: M[j,j] -= sum_k L[j,k]^2
    double sum = M[j * p + j];
    for (int k = 0; k < j; ++k) {
      const double Ljk = M[j * p + k];
      sum -= Ljk * Ljk;
    }

    if (sum <= 0.0)
      return -1e300; // Matrix not positive definite

    const double Ljj = std::sqrt(sum);
    M[j * p + j] = Ljj;
    logd += std::log(Ljj);

    // Off-diagonal: solve for column j below the diagonal
    const double invLjj = 1.0 / Ljj;
    for (int i = j + 1; i < p; ++i) {
      double s = M[i * p + j];
      for (int k = 0; k < j; ++k)
        s -= M[i * p + k] * M[j * p + k];
      M[i * p + j] = s * invLjj;
    }
  }

  // log|M| = 2 * sum log(diag(L))
  return 2.0 * logd;
}



// If you want to use LAPACK dpotrf when available, define USE_LAPACK_DPOTRF
// and link with your LAPACK/BLAS. Otherwise the hand-rolled Cholesky is used.
// #define USE_LAPACK_DPOTRF

#ifdef USE_LAPACK_DPOTRF
extern "C" {
  // dpotrf: Cholesky factorization of a real symmetric positive definite matrix.
  void dpotrf_(char* uplo, int* n, double* a, int* lda, int* info);
}
#endif

// -----------------------------------------------------------------------------
// Cholesky logdet on FULL covariance (no Gram):
// - M is (p x p) in row-major, symmetric; this routine will overwrite M with
//   the Cholesky factor (lower triangle) and junk in the upper.
// - Returns: log(det(M)) = 2 * sum(log(L_ii)).
// - If M is not SPD due to tiny negative round-off on diag or near-singularity,
//   a small diagonal ridge is added and the factorization is retried.
//
// -----------------------------------------------------------------------------
double Analysis_EntropyHD::LogDetChol_FullCov(double *M, int p)
{

    bool   scaleDiag  = true; // Whether to apply optional diagonal scaling for better conditioning.
    double ridge0     = 0.0;    // Initial ridge to add on failure (0 means auto-heuristic based on trace/p).
    int    maxRidgeIt = 5;     // Maximum number of ridge doubling attempts before giving up.

    double alpha = 46.03; // Schlitter fudge factor: hard-code for match to pczdump’s 300 * log(1 + 46.03 * λ_i) formula, where 46.03 = (2πe kB T / hbar^2) * (AMBER distance unit)^2
                       // Note that the fudge factor is applied as M = I + alpha * C, so alpha is the multiplier on the covariance matrix.
                       // The value of 46.03 corresponds to c_local = 46.03 in the Schlitter formula, which is what pczdump uses at T=300K with AMBER units.
                       // If you want to make this user-configurable, you could add a parameter for c_local and compute alpha = c_local / (n - 1) here.

    // M <- I + alpha*covariance
    for (int i = 0; i < p; ++i) {
        double* Ci = M + (size_t)i * p;
        for (int j = 0; j < p; ++j) 
             Ci[j] *= alpha;
        Ci[i] += 1.0; // add identity
    }


    // (1) Optional diagonal scaling: M = D^{-1/2} * M * D^{-1/2},
    //     where D is diag of M. Improves conditioning; we adjust logdet later.
    double scaleShift = 0.0; // add p * (-log s) to logdet if we scale by s per-row/col
    std::vector<double> dscale;
    if (scaleDiag) {
        dscale.resize(p, 1.0);
        // Build D^{-1/2} from the diagonal; handle zero/small diag defensively
        for (int i = 0; i < p; ++i) {
            const double di = M[(size_t)i * p + i];
            // If zero/negative from noise, clamp to tiny positive to proceed
            const double di_pos = (di > 0.0 ? di : 1e-300);
            dscale[i] = 1.0 / std::sqrt(di_pos);
        }
        // M <- D^{-1/2} * M * D^{-1/2}
        for (int i = 0; i < p; ++i) {
            double* Mi = M + (size_t)i * p;
            const double si = dscale[i];
            for (int j = 0; j < p; ++j) {
                Mi[j] *= si * dscale[j];
            }
        }
        // Adjust to later undo: det(M_old) = det(D^{1/2} M_new D^{1/2})
        // => log det(M_old) = log det(M_new) + sum_i log(di)  (since D = diag(di))
        // But we used di_pos; track correction using original diag or di_pos
        double sumLogD = 0.0;
        for (int i = 0; i < p; ++i) {
            const double di = 1.0 / (dscale[i] * dscale[i]); // = di_pos
            sumLogD += std::log(di);
        }
        // det scaling across both sides contributes +sumLogD
        scaleShift = sumLogD;
    }

    // (3) Prepare ridge heuristic if not provided
    double ridge = ridge0;
    if (ridge <= 0.0) {
        // Start ridge as tiny fraction of trace/p
        double tr = 0.0;
        for (int i = 0; i < p; ++i) tr += M[(size_t)i * p + i];
        double avg = (tr > 0.0 ? tr / (double)p : 1.0);
        ridge = std::max(1e-12, 1e-12 * avg);
    }

    // We will try factorization; on failure, add ridge to the diagonal and retry.
    for (int attempt = 0; attempt <= maxRidgeIt; ++attempt) {

        // ---- Hand-rolled lower-triangular Cholesky in-place ----
        // Make a working copy so we can re-try with ridge without accumulating multiple times.
        std::vector<double> A((size_t)p * p);
        std::memcpy(A.data(), M, sizeof(double) * (size_t)p * p);
        if (attempt > 0) {
            for (int i = 0; i < p; ++i) A[(size_t)i*p + i] += ridge;
        }

        bool ok = true;
        double logd = 0.0;

        for (int j = 0; j < p && ok; ++j) {
            // Update diagonal: A[j,j] -= sum_k A[j,k]^2
            double sum = A[(size_t)j * p + j];
            for (int k = 0; k < j; ++k) {
                const double Ljk = A[(size_t)j * p + k];
                sum -= Ljk * Ljk;
            }
            if (!(sum > 0.0)) { ok = false; break; }
            const double Ljj = std::sqrt(sum);
            A[(size_t)j * p + j] = Ljj;
            logd += std::log(Ljj);

            // Compute below-diagonal column j: A[i,j] = (A[i,j] - sum_k A[i,k]*A[j,k]) / Ljj
            const double invLjj = 1.0 / Ljj;
            for (int i = j + 1; i < p; ++i) {
                double s = A[(size_t)i * p + j];
                for (int k = 0; k < j; ++k) {
                    s -= A[(size_t)i * p + k] * A[(size_t)j * p + k];
                }
                A[(size_t)i * p + j] = s * invLjj;
            }
        }

        if (ok) {
            // Success: copy factor back to M if you wish
            std::memcpy(M, A.data(), sizeof(double) * (size_t)p * p);
            return 2.0 * logd + (scaleDiag ? scaleShift : 0.0);
        }
        // Not SPD enough: increase ridge and retry
        ridge *= 10.0;
    }
    // If we are here, matrix could not be stabilized
    return -1e300; // signal failure
}












// -----------------------------------------------------------------------------
// Eigensolve-free Schlitter log-det using the frame-space Gram trick.
//
// Computes: ld = log det( I + c_local * C )
// with C the (unbiased) covariance of X (n frames × p dims, row-major).
//
// By Sylvester: det(I_p + c C) == det(I_n + (c/(n-1)) Xc Xc^T),
// where Xc is X centered along columns.
//
// Complexity: O(n^2 p) to build the Gram, then O(n^3) for Cholesky.
// This is a win whenever n << p (typical MD).
// -----------------------------------------------------------------------------
double Analysis_EntropyHD::SchlitterEntropy(const double* X, int n, int p, double c_local, double *masses ) const
{
  // Basic guards
  if (X == nullptr || n <= 1 || p <= 0 || c_local <= 0.0) return 0.0;

  // --- Step 1: column means (length p) ---
  double* mean = dalloc(p);
  if (!mean) return 0.0;

  // mean[j] = average over frames of X[i*p + j]
  for (int j = 0; j < p; ++j) mean[j] = 0.0;
  for (int i = 0; i < n; ++i) {
    const double* row = X + (size_t)i * p;
    for (int j = 0; j < p; ++j) mean[j] += row[j];
  }
  const double invN = 1.0 / (double)n;
  for (int j = 0; j < p; ++j) mean[j] *= invN;

  // --- Step 2: build A = I_n + alpha * Xc * Xc^T  (n x n SPD) ---
  // alpha = c_local / (n-1)   (same unbiased convention as ComputeCov)
  const double alpha = c_local / (double)(n - 1);

  mprintf("Building covariance\n");
  //try working with a normal covariance matrix
  double* C = dalloc((size_t)p * p);
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) {
      C[i * p + j] = 0.0;
    } 
  }
  for (int i = 0; i < n; ++i){ //loop over frames
    const double* row = X + (size_t)i * p;
    for (int j = 0; j < p; ++j) { //loop over DOF
      const double xij = row[j] - mean[j];
      for (int k = 0; k < p; ++k) { //loop over DOF
        const double xik = row[k] - mean[k];
        C[j * p + k] += xij * xik;
      }
    }
  }
  for (int i = 0; i < p; ++i)
    for (int j = i; j < p; ++j) {
      double v = C[i*p + j] * invN * masses[i] * masses[j]; // covariance with mass weighting
      C[i*p + j] = v;
      C[j*p + i] = v;
    } 

  #ifdef ENTROPYHD_DEBUG
  FILE *F = std::fopen("cov_debug.txt", "w");
  for (int i = 0; i < p; ++i) {
     for (int j = 0; j < p; ++j) {
      std::fprintf(F, "%.2f ", C[i*p + j]);
     }
      std::fprintf(F, "\n");
  }
  std::fclose(F);
  #endif

  // --- Step 3: log-det via Cholesky ---
  // If numerical issues, add a tiny ridge to the diagonal and retry.
  //double ld = LogDetChol(A, n);

  double* M = (double*)malloc(sizeof(double) * (size_t)P_ * P_);

  
  //mprintf("Getting log determinant of I + Cov by Jacobi\n");
  //std::memcpy(M, C, sizeof(double) * (size_t)P_ * P_);
  //double ldJ = LogDetJac(M, p, c_local);
  
  mprintf("Getting log determinant of I + Cov by Cholesky\n");
  std::memcpy(M, C, sizeof(double) * (size_t)P_ * P_);
  double ld  = LogDetChol_FullCov(M, p);

  free(M);
  mprintf("DEBUG: Schlitter logdet via Cholesky = %.15e\n", ld);
  //mprintf("                          via Jacobi = %.15e\n", ldJ);

  // Cleanup
  std::free(mean);

  // ld == sum_i log(1 + c_local * lambda_i) == Schlitter "core" quantity
  return ld;
}

// Fit S(n) = S0 + m n^{-a}
void Analysis_EntropyHD::LMFitBiasModel(
  const double* nvals, const double* Svals, int K,
  double& S0, double& m, double& a, double& SE
) {
  // Initial guesses
  S0 = Svals[K - 1];
  m  = (K >= 2 ? (Svals[0] - S0) : 0.0);
  a  = 0.70;

  // LM parameters
  double lambda = 1e-3;
  const int MAXIT = 80;
  const double TOL = 1e-10;

  double prevRSS = 1e300;

  for (int it = 0; it < MAXIT; ++it) {
    double JtJ[9] = {0.0}, JtR[3] = {0.0};
    double rss = 0.0;

    // Build normal equations
    for (int k = 0; k < K; ++k) {
      const double n   = nvals[k];
      const double nmA = std::pow(n, -a);
      const double pred = S0 + m * nmA;
      const double r   = Svals[k] - pred;
      rss += r * r;

      // Jacobian wrt (S0, m, a)
      const double dS0 = -1.0;
      const double dm  = -nmA;
      const double da  = -(m * nmA * (-std::log(n)));

      // Accumulate J^T J
      JtJ[0] += dS0 * dS0;  JtJ[1] += dS0 * dm;  JtJ[2] += dS0 * da;
      JtJ[3] += dm  * dS0;  JtJ[4] += dm  * dm;  JtJ[5] += dm  * da;
      JtJ[6] += da  * dS0;  JtJ[7] += da  * dm;  JtJ[8] += da  * da;

      // Accumulate J^T r
      JtR[0] += dS0 * r;
      JtR[1] += dm  * r;
      JtR[2] += da  * r;
    }

    // LM damping
    JtJ[0] += lambda; JtJ[4] += lambda; JtJ[8] += lambda;

    // Solve 3x3 system for updates (Cramer's rule / adjoint)
    const double A=JtJ[0], B=JtJ[1], C=JtJ[2];
    const double D=JtJ[3], E=JtJ[4], F=JtJ[5];
    const double G=JtJ[6], H=JtJ[7], I=JtJ[8];

    const double det = A*(E*I - F*H) - B*(D*I - F*G) + C*(D*H - E*G);
    if (std::fabs(det) < 1e-18) break;

    const double dx0 =
      ( JtR[0]*(E*I - F*H) - B*(JtR[1]*I - F*JtR[2]) + C*(JtR[1]*H - E*JtR[2]) ) / det;
    const double dx1 =
      ( A*(JtR[1]*I - F*JtR[2]) - JtR[0]*(D*I - F*G) + C*(D*JtR[2] - JtR[1]*G) ) / det;
    const double dx2 =
      ( A*(E*JtR[2] - JtR[1]*H) - B*(D*JtR[2] - JtR[1]*G) + JtR[0]*(D*H - E*G) ) / det;

    // Try update
    const double newS0 = S0 + dx0;
    const double newm  = m  + dx1;
    const double newa  = a  + dx2;

    // Optional clamping for robustness
    double a_clamped = newa;
    if (a_clamped < 0.10) a_clamped = 0.10;
    if (a_clamped > 2.00) a_clamped = 2.00;

    // Evaluate RSS after tentative step (cheap recompute)
    double newRSS = 0.0;
    for (int k = 0; k < K; ++k) {
      const double n   = nvals[k];
      const double nmA = std::pow(n, -a_clamped);
      const double pred= newS0 + newm * nmA;
      const double r   = Svals[k] - pred;
      newRSS += r * r;
    }

    // LM accept/reject
    if (newRSS < rss) {
      S0 = newS0; m = newm; a = a_clamped;
      if (std::fabs(prevRSS - newRSS) < TOL) break;
      prevRSS = newRSS;
      lambda *= 0.5;               // relax damping
      if (lambda < 1e-12) lambda = 1e-12;
    } else {
      lambda *= 5.0;               // increase damping
      if (lambda > 1e6) break;
    }
  }

  // Standard error of S0 from covariance matrix
  // Rebuild J^T J (without damping) at final params
  double JtJ[9] = {0.0};
  double rss = 0.0;
  for (int k = 0; k < K; ++k) {
    const double n   = nvals[k];
    const double nmA = std::pow(n, -a);
    const double pred= S0 + m * nmA;
    const double r   = Svals[k] - pred;
    rss += r * r;

    const double dS0 = -1.0;
    const double dm  = -nmA;
    const double da  = -(m * nmA * (-std::log(n)));

    JtJ[0] += dS0 * dS0;  JtJ[1] += dS0 * dm;  JtJ[2] += dS0 * da;
    JtJ[3] += dm  * dS0;  JtJ[4] += dm  * dm;  JtJ[5] += dm  * da;
    JtJ[6] += da  * dS0;  JtJ[7] += da  * dm;  JtJ[8] += da  * da;
  }

  // (J^T J)^{-1}_{00}
  const double A=JtJ[0], B=JtJ[1], C=JtJ[2];
  const double D=JtJ[3], E=JtJ[4], F=JtJ[5];
  const double G=JtJ[6], H=JtJ[7], I=JtJ[8];
  const double DET = A*(E*I - F*H) - B*(D*I - F*G) + C*(D*H - E*G);
  const double cof00 = (E*I - F*H);

  const double dof = (double)(K - 3);
  const double sigma2 = (dof > 0.0 ? rss / dof : 0.0);
  const double varS0  = (DET != 0.0 ? sigma2 * (cof00 / DET) : 0.0);
  SE = (varS0 > 0.0 ? std::sqrt(varS0) : 0.0);
}

// -------------------------------- Analyze ------------------------------------
Analysis::RetType Analysis_EntropyHD::Analyze() {
  // ---------------------------------------
  // 1. Validate coordinate data
  // ---------------------------------------
  if (Coords_ == nullptr) {
    mprinterr("entropy_hd: No coordinate DataSet loaded.\n");
    return Analysis::ERR;
  }

  const Topology& top = Coords_->Top(); // Get topology reference from coordinates dataset... sometimes this is passed as a pointer....
  const int N      = Coords_->Size();   // number of frames

  mprintf("entropy_hd: running on coordinate set '%s' with %zu frames and %u dimensions.\n",
            coordsName_.c_str(), Coords_->Size(), Coords_->Ndim());
  

  if (N < window_) {
    mprinterr("entropy_hd: Not enough frames (%d) for window=%d.\n", N, window_);
    return Analysis::ERR;
  }

  // ---------------------------------------
  // 2. Setup atom mask
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
  std::vector<double> massVec(P_);   // p = 3*nSelected_
  for (int si = 0; si < nSelected_; si++) {
    int atom = mask_[si];
    double m = top[atom].Mass();          // mass in amu

    // Basic sanity check for hydrogens
    if (top[atom].AtomicNumber() == 1 && m > 2.01) {
        mprinterr("entropy_hd: Hydrogen atom mass %.2f > 2.01; suspect topology?\n", m);
        return Analysis::ERR;
    }
    // Each atom contributes three coordinate entries
    massVec[3*si + 0] = std::sqrt(m);
    massVec[3*si + 1] = std::sqrt(m);
    massVec[3*si + 2] = std::sqrt(m);
  }

  // ---------------------------------------
  // 3) Stage selected coordinates into memory,
  // align and centre
  // ---------------------------------------
  mprintf("allocating memory for coordinate matrix X with %d rows and %d columns...\n", N, P_);
  double* X = dalloc((size_t)N * P_);
  if (X == nullptr) {
    mprinterr("entropy_hd: Failed to allocate memory for coordinate matrix.\n");
    return Analysis::ERR;
  }
    BuildAlignedCoordinates(Coords_, mask_, X);
  

  // ---------------------------------------
  // 4. Compute Schlitter constant c(T) in AMBER units
  // ---------------------------------------
  static constexpr double kB_kcal     = 0.00198720425864083;
  static constexpr double hbar_kcalps = 0.0635078;
  const double c_local = (kB_kcal * temp_) / (hbar_kcalps * hbar_kcalps);

  // ---------------------------------------
  // 5. Compute number of windows
  // ---------------------------------------
  int K = N / window_;
 
  mprintf("using %d windows of size %d\n", K, window_);
  if ( K < 3 ) {
    mprinterr("entropy_hd: Not enough frames (%d) for window=%d to produce at least 3 windows; need at least three.\n", N, window_);
    std::free(X);
    return Analysis::ERR;
  }

  std::vector<double> nvals(K), Svals(K);

  // ---------------------------------------
  // 6. Compute S(n) for each prefix
  // ---------------------------------------
  for (int i = 0; i < K; ++i) {
    int n = (i + 1) * window_;

    mprintf("doing schlitter for window %d with %d frames\n", i + 1, n);

    Svals[i] = SchlitterEntropy( X, n, P_, c_local, massVec.data()) * (0.5* kB_kcal * temp_);  // free-energy units (kcal/mol);
    nvals[i] = (double)n;

    mprintf("window %d: n=%d frames, S(n)=%.4f kcal/mol @ T=%.1f K\n", i + 1, n, Svals[i], temp_);

  }

  // ---------------------------------------
  // 7. LM fit for each n_i and write K time points to DataSet
  // ---------------------------------------
  outputDS_->Allocate2D((size_t)K, 5);
  for (int i = 0; i < K; ++i) {
    int Ki = i + 1;

    double S0 = 0.0, m = 0.0, a = 0.0, SE = 0.0;
    LMFitBiasModel(nvals.data(), Svals.data(), Ki, S0, m, a, SE);

    double CIlo = S0 - 1.96*SE;
    double CIhi = S0 + 1.96*SE;

    outputDS_->SetElement(i, 0, nvals[i]);
    outputDS_->SetElement(i, 1, Svals[i]);
    outputDS_->SetElement(i, 2, S0);
    outputDS_->SetElement(i, 3, CIlo);
    outputDS_->SetElement(i, 4, CIhi);
    mprintf("ongoing estimate of configurational entropy at window %i: S0 = %.4f with 95%% CI (%.4f, %.4f)\n", 
       i, outputDS_->GetElement(i, 2), outputDS_->GetElement(i, 3), outputDS_->GetElement(i, 4));
  }
  mprintf("final estimate of configurational entropy S0 = %.4f with 95%% CI (%.4f, %.4f)\n", 
       outputDS_->GetElement(K-1, 2), outputDS_->GetElement(K-1, 3), outputDS_->GetElement(K-1, 4));

  // ---------------------------------------
  // 8. Optional ASCII output
  // ---------------------------------------
  if (!outfileName_.empty()) {
    FILE* F = fopen(outfileName_.c_str(), "w");
    fprintf(F, "##Harris-Dryden entropy estimator: \n");
    if (F) {
      fprintf(F, "# n   S(n)   S0(n)   CIlo   CIhi\n");
      for (int i = 0; i < K; ++i)
        fprintf(F, "%d %.12f %.12f %.12f %.12f\n",
                (int)nvals[i],
                outputDS_->GetElement(i,1), outputDS_->GetElement(i,2),
                outputDS_->GetElement(i,3), outputDS_->GetElement(i,4));
      fclose(F);
    }
  }

  // ---------------------------------------
  // 9. Optional save out detailed report
  // ---------------------------------------
  if (!detailsName_.empty()) {
    FILE* F = fopen(detailsName_.c_str(), "w");
    if (F) {
      fprintf(F, "#Harris-Dryden unbiased entropy estimator given time series of Molecular Dynamics Frames:\n");
      fprintf(F, "#References:\n");
      fprintf(F, "#   - Dryden, Kume, Le, Wood (Stat. inference for functions of covariance...,\n");
      fprintf(F, "#     Sections 4.1–4.2, Schlitter entropy and bias behavior).  \n");
      fprintf(F, "#   - Harris et al. (2001) empirical bias scaling for S(nr S(n). \n");
      fprintf(F, "#Mask: %s\n", maskString_.empty() ? "<all atoms>" : maskString_.c_str());
      fprintf(F, "#Selected atoms: %d (P=%d)\n", nSelected_, P_);
      fprintf(F, "#Temp=%.1f K  window=%d  frames=%d\n\n", temp_, window_, N);
      fprintf(F, "#n     S(n)        S0(n)       CIlo        CIhi\n");
      for (int i = 0; i < K; ++i)
        fprintf(F, "%d %.6f %.6f %.6f %.6f\n",
                (int)nvals[i],
                outputDS_->GetElement(i,1), outputDS_->GetElement(i,2),
                outputDS_->GetElement(i,3), outputDS_->GetElement(i,4));

      fclose(F);
    }
  }

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

    // First-pass mean (Cartesian), second-pass mean 
    std::vector<double> avg1(p, 0.0);

    Frame ref, fr, avgFr;
    ref.SetupFrameFromMask(mask_);
    fr.SetupFrameFromMask(mask_);
    avgFr.SetupFrameFromMask(mask_);

    // -----------------------------------------------------------
    // PASS 1: align to frame 0 and accumulate Cartesian average
    // -----------------------------------------------------------
    Coords_->GetFrame(0, ref);

    for (int f = 0; f < nFrames; f++) {
        Coords_->GetFrame(f, fr);
        fr.Align(ref, mask_);

        for (int si = 0; si < nSel; si++) {
            const double* xyz = fr.XYZ(si);
            avg1[3*si + 0] += xyz[0];
            avg1[3*si + 1] += xyz[1];
            avg1[3*si + 2] += xyz[2];
        }
    }

    const double invN = 1.0 / double(nFrames);
    for (int j = 0; j < p; j++) avg1[j] *= invN;

    // -----------------------------------------------------------
    // Build average frame from first-pass mean (Cartesian)
    // -----------------------------------------------------------
    {
        std::vector<double> dummyMass(nSel, 1.0);
        avgFr.SetupFrameXM(avg1, dummyMass);
    }

    // -----------------------------------------------------------
    // PASS 2: align to avgFr, then:
    //   - remove translation (COM),
    //   - remove rotation via projection (3 rotational modes),
    //   - store in Xmw.
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
            Coords_->GetFrame(f, local);
            local.Align(avgFr, mask_);

            // ---------- (1) Translation removal:  COM ----------
            // COM = sum(m_i * x_i) / sum(m_i)
            double comX = 0.0, comY = 0.0, comZ = 0.0;
            for (int si = 0; si < nSel; si++) {
                const double* xyz = local.XYZ(si);
                comX +=  xyz[0];
                comY +=  xyz[1];
                comZ +=  xyz[2];
            }
            comX /= nSel; comY /= nSel; comZ /= nSel;
            

            // ---------- (2) Build centered y and rotational bases Bx,By,Bz ----------
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

            // ---------- (3) Build Gram G = B^T B and rhs = B^T y (3x3 and 3x1) ----------
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

            // ---------- (4) Solve G c = rhs (3x3) with tiny Tikhonov regularization ----------
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

            // ---------- (5) Project rotation: y <- y - (c0*Bx + c1*By + c2*Bz) ----------
            for (int j = 0; j < p; j++)
                y[j] = y[j] - (c[0]*Bx[j] + c[1]*By[j] + c[2]*Bz[j]);

            // ---------- (6) Store RB-removed coordinates ----------
            double* row = X + (size_t)f * p;
            for (int j = 0; j < p; j++) row[j] = y[j];
        }
    }

    // -----------------------------------------------------------
    // SECOND-PASS MEAN and CENTERING
    // -----------------------------------------------------------
    std::fill(avg1.begin(), avg1.end(), 0.0);

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
            avg1[j] += threadSum[t][j];

    for (int j = 0; j < p; j++)
        avg1[j] *= invN;

#pragma omp parallel for schedule(static)
    for (int f = 0; f < nFrames; f++) {
        double* row = X + (size_t)f * p;
        for (int j = 0; j < p; j++)
            row[j] -= avg1[j];
    }

    mprintf("Aligned, rigid-body removed, and centered: %d frames × %d dims.\n",
            nFrames, p);
}


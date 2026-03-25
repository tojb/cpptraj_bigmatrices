#ifndef INC_ANALYSIS_ENTROPYHD_H
#define INC_ANALYSIS_ENTROPYHD_H

#include "Analysis.h"
#include "ArgList.h"
#include "DataSet_MatrixDbl.h"
#include "DataSet_Coords.h"
#include "AtomMask.h"
#include <string>

class Analysis_EntropyHD : public Analysis {
public:
  Analysis_EntropyHD();
  DispatchObject* Alloc() const { return (DispatchObject*)new Analysis_EntropyHD(); }

  void Help() const override;

  // IMPORTANT: Use AnalysisSetup& per cpptraj Analysis modules.
  Analysis::RetType Setup(ArgList& al, AnalysisSetup& setup, int debugIn);
  Analysis::RetType Analyze();

  //Expose this function in case someone wants it.
  void BuildAlignedCoordinates(
        DataSet_Coords* Coords_,
        const AtomMask& mask_,
              double* X );

private:
  std::string dsoutName_;
  std::string coordsName_;
  std::string maskString_;
  std::string outfileName_;
  std::string detailsName_;
  double      temp_;
  int         window_;

  DataSet_Coords*     Coords_;
  DataSet_MatrixDbl*  outputDS_;
  AtomMask            mask_;
  int                 nSelected_;
  size_t              P_;

  static void   ComputeMean(double* mean, const double* X, int n, int p);
  void          ComputeCov (double* C, const double* X, const double* mean, int n );
  double        LogDetChol_FullCov_Broken ( double* M ) const;
  double        LogDetChol_FullCov ( double* M ) const;
  double        LogDetChol_GramOrFullCov ( double* M, const double *X, double *masses, size_t n ) const;
  double        LogDetChol ( double* M );
  double        LogDetJac (double* M, size_t n ) const;
  double        SchlitterEntropy(const double* X, int n, double *masses) const;
  double        Stable_Schlitter_LogDet(const double* Xc, const size_t n, const size_t p, const double *masses) const;

  static void   LMFitBiasModel(const double* nvals, const double* Svals, int K,
                                     double& S0, double& m, double& a, double& SE);

  static inline double* dalloc(size_t n) { 
	  double* p = (double*) std::malloc(n * sizeof(double));
	  if(!p) mprinterr("memory fail in AnalysisEntropy!!\n");
	  return p; 
  }
};

#endif

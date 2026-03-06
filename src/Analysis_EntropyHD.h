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
  int                 P_;

  static void   ComputeMean(double* mean, const double* X, int n, int p);
  static void   ComputeCov (double* C, const double* X, const double* mean, int n, int p);
  static double LogDetChol_FullCov ( double* M, int p );
  static double LogDetChol (double* M, int p);
  static double LogDetJac (double* M, int p, const double c_local);
  double        SchlitterEntropy(const double* X, int n, int p, double c_local, double *masses) const;
  double        _SchlitterEntropy(const double* X, int n, int p, double c_local) const;
  static void   LMFitBiasModel(const double* nvals, const double* Svals, int K,
                                     double& S0, double& m, double& a, double& SE);

  static inline double* dalloc(size_t n) { return (double*) std::malloc(n * sizeof(double)); }
};

#endif
#ifndef INC_ANALYSIS_REGRESSION_H
#define INC_ANALYSIS_REGRESSION_H


/* -----------------------------------------------------------------------------
 *  This file is part of cpptraj (AmberTools).
 *
 *  Purpose:
 *    Modular regression analysis with pluggable models (linear, exp, power,
 *    stretched-exponential, logistic, Michaelis–Menten) and Levenberg–Marquardt
 *    parameter estimation, including standard errors and covariance output.
 *
 *  Provenance:
 *    Based on (and extending) the original Analysis_Regression implementation
 *    and Analysis module patterns in cpptraj. The regression code in this file
 *    replaces the previous linear-only variant with a general model registry
 *    and LM optimizer.
 *
 *  Attribution:
 *    The overall Analysis framework, dataset interfaces, and I/O patterns are
 *    as usual in cpptraj. The nonlinear least-squares routine follows the
 *    classical Levenberg–Marquardt approach (Levenberg, 1944; Marquardt, 1963).
 *
 *  License:
 *    Distributed under the same license terms as the rest of cpptraj. See the
 *    cpptraj LICENSE file in the repository root for details.
 *
 *  Authors:
 *    Josh Berryman <josh.berryman@uni.lu>
 *
 *    With acknowledgements to the AMBER community and the
 *    *many* cpptraj authors and contributors.
 * --------------------------------------------------------------------------- */


#include "Analysis.h"
#include "Array1D.h"
#include "ArgList.h"
#include "DataSet_1D.h"
#include "DataSet.h"
#include "DataFile.h"
#include "DataSetList.h"
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>

//convenience structure for storing regression outputs.
struct RegressionParamSetHandles {
    std::vector<DataSet*> values;  // one DOUBLE dataset per parameter (e.g., slope/intercept)
    std::vector<DataSet*> errors;  // one DOUBLE dataset per parameter error (e.g., slope_err)
    bool ok() const { return !values.empty(); } // simple sanity check
};


// ============================================================================
//  REGRESSION MODEL FRAMEWORK (local to this module)
//
//  Developers: you should be able to add new models fairly easily by 
//  implementing this class, copy e.g. ExpSatModel() or other non-trivial
//  function fit.
//
// ============================================================================

// --------------------------- RegressionModelBase ----------------------------
class RegressionModelBase {
public:
  virtual ~RegressionModelBase() {}

  // Number of parameters.
  virtual int NumParams() const = 0;

  // f(x; p)
  virtual double Evaluate(double x, const std::vector<double>& p) const = 0;

  // Jacobian df/dp_k (fills J with size NumParams()).
  virtual void Jacobian(double x,
                        const std::vector<double>& p,
                        std::vector<double>& J) const = 0;

  // Heuristic initial guess from data.
  virtual void InitialGuess(const std::vector<double>& x,
                            const std::vector<double>& y,
                            std::vector<double>& p) const = 0;

  // Parameter names for formatted output.
  virtual std::vector<std::string> ParamNames() const = 0;
};

// ---------------------------- LinearModel -----------------------------------
class LinearModel : public RegressionModelBase {
	/*
	 * This class fits a straight line.
	 */
public:
  int NumParams() const override { return 2; }

  double Evaluate(double x, const std::vector<double>& p) const override {
    return p[0] + p[1] * x; // A + B x
  }

  void Jacobian(double x, const std::vector<double>& p,
                std::vector<double>& J) const override {
    J[0] = 1.0; // d/dA
    J[1] = x;   // d/dB
  }

  void InitialGuess(const std::vector<double>& x,
                    const std::vector<double>& y,
                    std::vector<double>& p) const override {
    p.resize(2);
    if (x.size() >= 2 && x.back() != x.front()) {
      const double slope = (y.back() - y.front()) / (x.back() - x.front());
      p[1] = slope;
      p[0] = y.front() - slope * x.front();
    } else {
      p[0] = y.front();
      p[1] = 0.0;
    }
  }

  std::vector<std::string> ParamNames() const override {
    return {"intercept", "slope"};
  }
};

// ------------------------ Exponential Saturation Model ----------------------
class ExpSatModel : public RegressionModelBase {
	/*
         * This class fits f(x)=A(1-exp(-B*x))
	 *
	 * A and B are initialised as positive.
	 *
	 * The intention here is to model the growth of (apparent) entropy
	 * versus sampling time for an MD trajectory, in particular
	 * this supports entropy analysis via the Schlitter method.
	 *
         */
public:
  int NumParams() const override { return 2; }

  double Evaluate(double t, const std::vector<double>& p) const override {
    const double A = p[0], B = p[1];
    return A * (1.0 - std::exp(-B * t));
  }

  void Jacobian(double t, const std::vector<double>& p,
                std::vector<double>& J) const override {
    const double A = p[0], B = p[1];
    const double e = std::exp(-B * t);
    J[0] = 1.0 - e;      // d/dA
    J[1] = A * t * e;    // d/dB
  }

  void InitialGuess(const std::vector<double>& x,
                    const std::vector<double>& y,
                    std::vector<double>& p) const override {
    p.assign(2, 0.0);
    p[0] = y.back();                     // asymptote ~ last value
    const double span = std::max(1e-12, x.back() - x.front());
    p[1] = 1.0 / (0.3 * span);           // mild rate guess
  }

  std::vector<std::string> ParamNames() const override { return {"A", "B"}; }
};

// ---------------------------- Power-Law Model -------------------------------
class PowerLawModel : public RegressionModelBase {
	/*
	 * This class fits f(x) = A*x**B
	 */
public:
  int NumParams() const override { return 2; }

  double Evaluate(double t, const std::vector<double>& p) const override {
    const double A = p[0], B = p[1];
    const double tp = (t > 0.0 ? t : 1e-12);
    return A * std::pow(tp, B);
  }

  void Jacobian(double t, const std::vector<double>& p,
                std::vector<double>& J) const override {
    const double A = p[0], B = p[1];
    const double tp = (t > 0.0 ? t : 1e-12);
    const double tb = std::pow(tp, B);
    J[0] = tb;                 // d/dA
    J[1] = A * tb * std::log(tp); // d/dB
  }

  void InitialGuess(const std::vector<double>& x,
                    const std::vector<double>& y,
                    std::vector<double>& p) const override {
    p.assign(2, 0.0);
    p[0] = std::max(1e-12, y.front());
    p[1] = 1.0;
  }

  std::vector<std::string> ParamNames() const override { return {"A", "B"}; }
};

// ------------------------- Stretched Exponential ----------------------------
class StretchedExpModel : public RegressionModelBase {
	/*
	 * This class fits y = A * (1 - exp( -(B t)^C ))
	 */
public:
  int NumParams() const override { return 3; }

  // y = A * (1 - exp( -(B t)^C ))
  double Evaluate(double t, const std::vector<double>& p) const override {
    const double A = p[0], B = p[1], C = p[2];
    const double tp = std::max(0.0, t);
    const double z = std::pow(B * tp, C);
    return A * (1.0 - std::exp(-z));
  }

  void Jacobian(double t, const std::vector<double>& p,
                std::vector<double>& J) const override {
    const double A = p[0], B = p[1], C = p[2];
    const double tp = std::max(0.0, t);
    const double Bt = B * tp;
    const double epsilon = 1e-12;
    const double Bt_pos = std::max(Bt, epsilon);
    const double z = std::pow(Bt_pos, C); // z = (B t)^C
    const double e = std::exp(-z);

    // df/dA = (1 - e^{-z})
    J[0] = (1.0 - e);

    // df/dB = A * e^{-z} * d(z)/dB  where z = (B t)^C
    // d(z)/dB = C * (B t)^{C-1} * t
    const double dz_dB = (C * std::pow(Bt_pos, C - 1.0)) * tp;
    J[1] = A * e * dz_dB;

    // df/dC = A * e^{-z} * d(z)/dC
    // d(z)/dC = (B t)^C * ln(B t)
    const double dz_dC = z * std::log(Bt_pos);
    J[2] = A * e * dz_dC;
  }

  void InitialGuess(const std::vector<double>& x,
                    const std::vector<double>& y,
                    std::vector<double>& p) const override {
    p.assign(3, 0.0);
    p[0] = y.back();                                // A ~ final plateau
    const double span = std::max(1e-12, x.back() - x.front());
    p[1] = 1.0 / (0.4 * span);                      // B ~ rate scale
    p[2] = 0.8;                                     // C ~ stretch
  }

  std::vector<std::string> ParamNames() const override { return {"A","B","C"}; }
};

// ------------------------------- Logistic -----------------------------------
class LogisticModel : public RegressionModelBase {
        /*
         * This class fits the logistic function:
	 *
	 *     y = A / (1 + exp(-B (x - C)))
	 *
         */
public:
  int NumParams() const override { return 3; }

  // y = A / (1 + exp(-B (x - C)))
  double Evaluate(double x, const std::vector<double>& p) const override {
    const double A = p[0], B = p[1], C = p[2];
    const double z = -B * (x - C);
    return A / (1.0 + std::exp(z));
  }

  void Jacobian(double x, const std::vector<double>& p,
                std::vector<double>& J) const override {
    const double A = p[0], B = p[1], C = p[2];
    const double z = -B * (x - C);
    const double e = std::exp(z);
    const double denom = (1.0 + e);
    const double denom2 = denom * denom;

    // f = A / denom
    // df/dA = 1/denom
    J[0] = 1.0 / denom;

    // df/dB = A * d(1/denom)/dB = -A * ( - (x - C) * e ) / denom^2
    J[1] = A * (x - C) * e / denom2;

    // df/dC = A * d(1/denom)/dC = -A * ( (B) * e ) / denom^2
    J[2] = -A * B * e / denom2;
  }

  void InitialGuess(const std::vector<double>& x,
                    const std::vector<double>& y,
                    std::vector<double>& p) const override {
    p.assign(3, 0.0);
    p[0] = std::max(y.front(), y.back());           // A ~ upper plateau
    p[1] = 1.0 / std::max(1e-12, (x.back() - x.front())/4.0); // slope scale
    p[2] = 0.5 * (x.front() + x.back());            // midpoint
  }

  std::vector<std::string> ParamNames() const override { return {"A","B","C"}; }
};

// ------------------------- Michaelis–Menten (mm) ----------------------------
class MichaelisMentenModel : public RegressionModelBase {
    /*
     *  Michaelis-Menten is a common model for reaction kinetics
     *
     *  y = A x / (B + x)
     *
     */
public:
  int NumParams() const override { return 2; }

  double Evaluate(double x, const std::vector<double>& p) const override {
    const double A = p[0], B = p[1];
    return A * x / (B + std::max(1e-12, x));
  }

  void Jacobian(double x, const std::vector<double>& p,
                std::vector<double>& J) const override {
    const double A = p[0], B = p[1];
    const double xp = std::max(1e-12, x);
    const double denom = (B + xp);
    J[0] = xp / denom;                    // d/dA
    J[1] = -A * xp / (denom * denom);     // d/dB
  }

  void InitialGuess(const std::vector<double>& x,
                    const std::vector<double>& y,
                    std::vector<double>& p) const override {
    p.assign(2, 0.0);
    p[0] = y.back();                      // Vmax
    p[1] = 0.1 * std::max(1e-12, x.back());// Km
  }

  std::vector<std::string> ParamNames() const override { return {"A","B"}; }
};

// ============================================================================
//  Model Registry (local to this module)
// ============================================================================
class ModelRegistry {
	/*
	 *  Build a list of models available for regression.
	 *
	 */
public:
  using Factory = RegressionModelBase*(*)();

  static ModelRegistry& Instance() {
    static ModelRegistry R;
    return R;
  }

  void Register(const std::string& name, Factory f) { table_[name] = f; }

  RegressionModelBase* Create(const std::string& name) const {
    auto it = table_.find(name);
    return (it != table_.end() ? it->second() : nullptr);
  }

  std::vector<std::string> List() const {
    std::vector<std::string> v;
    for (auto& kv : table_) v.push_back(kv.first);
    return v;
  }

private:
  std::map<std::string,Factory> table_;
};

// ============================================================================
//  Levenberg–Marquardt optimizer (internal)
// ============================================================================
bool LM_Optimize(const RegressionModelBase& model,
                 const std::vector<double>& x,
                 const std::vector<double>& y,
                 std::vector<double>& p,
                 double& rss_out,
                 std::vector<double>& cov_out,    // (M x M) covariance
                 std::vector<double>& stderr_out, // (M) standard errors
		 int    maxit, 
		 double tol
                 );

// ============================================================================
//  Analysis_Regression (exports to cpptraj)
// ============================================================================
class Analysis_Regression : public Analysis {
public:
  Analysis_Regression();
  virtual void Help() const;
  virtual RetType Setup(ArgList&, AnalysisSetup&, int);
  virtual RetType Analyze();
  virtual DispatchObject* Alloc() const override { return new Analysis_Regression(); }

private:
  // Inputs
  Array1D     input_dsets_;
  Array1D     output_dsets_;
  std::string fitModel_;     // "linear","exp","power","stretch","logistic","mm"
  int nx_;                   // number of evaluation X-points (if >1)

  //optimisation parameters
  double lm_tol_;         // default e.g. 1e-12
  int    lm_maxit_;       // default e.g. 200

  // Outputs
  std::vector< std::vector<DataSet*> >  paramSets_;  // parameters of fitted curves: per model instance
  std::vector< std::vector<DataSet*> >  paramErrSets_; //Errorbars
  std::vector< std::string >            paramNames_; // names for each parameter: per model type (ie one only)

  CpptrajFile* statsout_;               // stats file
  DataFile*    outfile_;

  // Runtime
  RegressionModelBase* model_;          // chosen model
};

#endif


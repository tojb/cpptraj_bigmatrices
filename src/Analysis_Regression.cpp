/* -----------------------------------------------------------------------------
 * This file is part of cpptraj (AmberTools).
 *
 * Purpose:
 *   Modular regression analysis with pluggable models (linear, exp, power,
 *   stretched-exponential, logistic, Michaelis–Menten) and Levenberg–Marquardt
 *   parameter estimation, including standard errors and covariance output.
 *
 * Provenance:
 *   Based on (and extending) the original Analysis_Regression implementation
 *   and Analysis module patterns in cpptraj. The regression code in this file
 *   replaces the previous linear-only variant with a general model registry
 *   and LM optimizer.
 *
 * Attribution:
 *   The overall Analysis framework, dataset interfaces, and I/O patterns are
 *   as usual in cpptraj. The nonlinear least-squares routine follows the
 *   classical Levenberg–Marquardt approach (Levenberg, 1944; Marquardt, 1963).
 *
 * License:
 *   Distributed under the same license terms as the rest of cpptraj. See the
 *   cpptraj LICENSE file in the repository root for details.
 *
 * Authors:
 *   Josh Berryman <josh.berryman@uni.lu>
 *
 *   With acknowledgements to the AMBER community and the
 *   *many* cpptraj authors and contributors.
 * --------------------------------------------------------------------------- */

#include "Analysis_Regression.h"
#include "CpptrajStdio.h"
#include "DataSet_Mesh.h"
#include <cstdio>
#include <limits>

// ============================================================================
// Register built-in models
// ============================================================================
static RegressionModelBase* CreateLinear()          { return new LinearModel(); }
static RegressionModelBase* CreateExpSat()          { return new ExpSatModel(); }
static RegressionModelBase* CreatePower()           { return new PowerLawModel(); }
static RegressionModelBase* CreateStretch()         { return new StretchedExpModel(); }
static RegressionModelBase* CreateLogistic()        { return new LogisticModel(); }
static RegressionModelBase* CreateMichaelisMenten() { return new MichaelisMentenModel(); }

static bool register_models() {
  ModelRegistry::Instance().Register("linear",   CreateLinear);
  ModelRegistry::Instance().Register("expsat",   CreateExpSat);
  ModelRegistry::Instance().Register("power",    CreatePower);
  ModelRegistry::Instance().Register("stretch",  CreateStretch);
  ModelRegistry::Instance().Register("logistic", CreateLogistic);
  ModelRegistry::Instance().Register("mm",       CreateMichaelisMenten);
  return true;
}
static bool _models_registered = register_models();

// ============================================================================
// Small helpers
// ============================================================================
static inline void zero(std::vector<double>& v) { std::fill(v.begin(), v.end(), 0.0); }

static bool invert2x2(const double A[4], double invA[4]) {
  const double det = A[0]*A[3] - A[1]*A[2];
  if (std::fabs(det) < 1e-20) return false;
  const double invdet = 1.0/det;
  invA[0] =  A[3]*invdet;
  invA[1] = -A[1]*invdet;
  invA[2] = -A[2]*invdet;
  invA[3] =  A[0]*invdet;
  return true;
}

static bool invert3x3(const double A[9], double invA[9]) {
  const double a = A[0], b = A[1], c = A[2];
  const double d = A[3], e = A[4], f = A[5];
  const double g = A[6], h = A[7], i = A[8];

  const double A11 =  (e*i - f*h);
  const double A12 = -(d*i - f*g);
  const double A13 =  (d*h - e*g);
  const double A21 = -(b*i - c*h);
  const double A22 =  (a*i - c*g);
  const double A23 = -(a*h - b*g);
  const double A31 =  (b*f - c*e);
  const double A32 = -(a*f - c*d);
  const double A33 =  (a*e - b*d);

  const double det = a*A11 + b*A12 + c*A13;
  if (std::fabs(det) < 1e-24) return false;

  const double invdet = 1.0/det;
  invA[0] = A11*invdet; invA[1] = A21*invdet; invA[2] = A31*invdet;
  invA[3] = A12*invdet; invA[4] = A22*invdet; invA[5] = A32*invdet;
  invA[6] = A13*invdet; invA[7] = A23*invdet; invA[8] = A33*invdet;

  return true;
}


// Normalize parameter names to length M, using model names if provided,
// otherwise synthesize A,B,C,...  and apply legacy aliases for 'linear'.
static inline std::vector<std::string>
NormalizedParamNames(const  std::vector<std::string>& modelNames,
                     const  std::string&              fitModel,
		     size_t M)
{
    std::vector<std::string> names = modelNames;
    names.resize(M);

    //set default parameter names as A,B,C etc.
    for (size_t i = 0; i < M; ++i)
        if (names[i].empty()) names[i] = std::string(1, char('A' + int(i)));

    if (fitModel == "linear" && names.size() >= 2) {
        names[0] = "intercept";     // rename "A" if the model is a linear fit
        names[1] = "slope"; // "B" is called "slope". Reverse of normal convention.
    }
    return names;
}

// Build a structure contiaining  
// multiple pairs of one-scalar
// DataSet objects, each pair is a fit parameter and an error.
static inline RegressionParamSetHandles
CreateParamSetWithErrors(DataSetList&                    DSL,
                         const std::string&              baseName,
                         int                             idx,
                         const std::vector<std::string>& names,
                         const std::string&              publicLegend)
{
    RegressionParamSetHandles H;

    // Create value datasets
    H.values.reserve(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
	mprintf("DEBUG: creating a MetaData object %s, %s, %i\n", baseName.c_str(), names[i].c_str(), idx);
        DataSet* v = DSL.AddSet(DataSet::DOUBLE, MetaData(baseName, names[i], idx));
        if (!v) { H.values.clear(); H.errors.clear(); return H; }
        v->SetLegend(publicLegend+"["+names[i]+"]");
        H.values.push_back(v);
    }

    // Create error datasets with "<name>_err"
    H.errors.reserve(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        const std::string ename = names[i] + "_err";
        DataSet* e = DSL.AddSet(DataSet::DOUBLE, MetaData(baseName, ename.c_str(), idx));
        if (!e) { H.values.clear(); H.errors.clear(); return H; }
        e->SetLegend(publicLegend+"["+ename+"]");
        H.errors.push_back(e);
    }

    return H;
}

// Save some regression results into 
// the value & uncertainty fields of an existing 
// RegressionParamSetHandles object.
static inline void
FillParamSetsAndAttach(const std::vector<DataSet*>& valueSets,
                       const std::vector<double>&   params,
                       const std::vector<DataSet*>& errorSets,
                       const std::vector<double>&   stderrs,
                       DataFile* outfile) // may be null
{
    // Values
    const size_t nv = std::min(valueSets.size(), params.size());
    for (size_t i = 0; i < nv; ++i) {
        const double v = params[i];
	mprintf("attaching %s param value for output: %f\n", valueSets[i]->legend(), v);
        valueSets[i]->Add((size_t)0, &v);            // scalar DOUBLE: Add(index, &value)
        if (outfile != 0) outfile->AddDataSet(valueSets[i]);
    }

    // Errors (if present)
    const size_t ne = std::min(errorSets.size(), stderrs.size());
    for (size_t i = 0; i < ne; ++i) {
        const double e = stderrs[i];
        errorSets[i]->Add((size_t)0, &e);
        if (outfile != 0) outfile->AddDataSet(errorSets[i]);
    }
}


// ============================================================================
// Levenberg–Marquardt Optimizer with covariance/SE output
// - Supports 2- and 3-parameter models.
// ============================================================================
bool LM_Optimize(const RegressionModelBase& model,
                 const std::vector<double>& x,
                 const std::vector<double>& y,
                 std::vector<double>& p,
                 double& rss_out,
                 std::vector<double>& cov_out,
                 std::vector<double>& stderr_out,
		 int    maxit,
		 double tol)
{
  const int N = (int)x.size();
  const int M = model.NumParams();
  if (N < M) return false;

  double lambda    = 1e-3;

  std::vector<double> J(M), JtJ(M*M), JtR(M);
  std::vector<double> p_new(M);

  auto compute_rss = [&](const std::vector<double>& params)->double {
    double rss = 0.0;
    for (int i=0;i<N;i++) {
      const double r = y[i] - model.Evaluate(x[i], params);
      rss += r*r;
    }
    return rss;
  };

  // Ensure initial guess present:
  if ((int)p.size() != M) {
    p.assign(M, 0.0);
  }

  double prevRSS = compute_rss(p);

  for (int iter=0; iter<maxit; ++iter) {
    zero(JtJ);
    zero(JtR);

    double rss = 0.0;

    for (int i=0;i<N;i++) {
      const double xi = x[i], yi = y[i];
      const double pred = model.Evaluate(xi, p);
      const double r = yi - pred;

      model.Jacobian(xi, p, J);

      // J^T J
      for (int a=0;a<M;a++)
        for (int b=0;b<M;b++)
          JtJ[a*M + b] += J[a]*J[b];

      // J^T r
      for (int a=0;a<M;a++)
        JtR[a] += J[a]*r;

      rss += r*r;
    }

    // LM damping on diagonal
    for (int a=0;a<M;a++) JtJ[a*M + a] += lambda;

    // Solve (JtJ) dp = JtR
    bool ok=false;
    if (M==2) {
      double inv[4];
      ok = invert2x2(JtJ.data(), inv);
      if (!ok) break;
      p_new[0] = p[0] + inv[0]*JtR[0] + inv[1]*JtR[1];
      p_new[1] = p[1] + inv[2]*JtR[0] + inv[3]*JtR[1];
    } else if (M==3) {
      double inv[9];
      ok = invert3x3(JtJ.data(), inv);
      if (!ok) break;
      p_new[0] = p[0] + inv[0]*JtR[0] + inv[1]*JtR[1] + inv[2]*JtR[2];
      p_new[1] = p[1] + inv[3]*JtR[0] + inv[4]*JtR[1] + inv[5]*JtR[2];
      p_new[2] = p[2] + inv[6]*JtR[0] + inv[7]*JtR[1] + inv[8]*JtR[2];
    } else {
      // Only 2- and 3-parameter models supported here.
      return false;
    }

    double newRSS = compute_rss(p_new);
    if (newRSS < rss) {
      p = p_new;
      if (std::fabs(prevRSS - newRSS) < tol) {
        prevRSS = newRSS;
        break;
      }
      prevRSS = newRSS;
      lambda *= 0.5;
      if (lambda < 1e-15) lambda = 1e-15;
    } else {
      lambda *= 5.0;
      if (lambda > 1e15) break;
    }
  }

  // --- Covariance and standard errors at final parameters ---
  // Rebuild (J^T J) without LM damping
  zero(JtJ);
  double rss = 0.0;
  for (int i=0;i<N;i++) {
    const double r = y[i] - model.Evaluate(x[i], p);
    rss += r*r;
    model.Jacobian(x[i], p, J);
    for (int a=0;a<M;a++)
      for (int b=0;b<M;b++)
        JtJ[a*M + b] += J[a]*J[b];
  }

  // Cov = sigma^2 * (J^T J)^(-1), sigma^2 = RSS/(N-M)
  const double dof    = std::max(1, N - M);
  const double sigma2 = rss / (double)dof;

  cov_out.assign(M*M, 0.0);
  bool inv_ok = false;
  if (M==2) {
    double inv[4];
    inv_ok = invert2x2(JtJ.data(), inv);
    if (inv_ok) {
      for (int k=0;k<4;k++) cov_out[k] = sigma2 * inv[k];
    }
  } else if (M==3) {
    double inv[9];
    inv_ok = invert3x3(JtJ.data(), inv);
    if (inv_ok) {
      for (int k=0;k<9;k++) cov_out[k] = sigma2 * inv[k];
    }
  }

  stderr_out.assign(M, 0.0);
  if (inv_ok) {
    for (int a=0;a<M;a++) {
      const double v = cov_out[a*M + a];
      stderr_out[a] = (v > 0.0 ? std::sqrt(v) : 0.0);
    }
  }

  rss_out = rss;
  return true;
}

// ============================================================================
// Analysis_Regression
// ============================================================================
Analysis_Regression::Analysis_Regression()
  : input_dsets_(),
    output_dsets_(),
    fitModel_(),
    nx_(0),
    lm_tol_(1e-12),
    lm_maxit_(200),
    statsout_(0),
    outfile_(0),
    model_(0)
{}

void Analysis_Regression::Help() const {
  mprintf("analysis regression <dset0> [<dset1> ...]\n"
          "  [model <linear|exp|power|stretch|logistic|mm>]\n"
	  "  [tol <lm_tol>] [maxit <lm_maxit>]\n"
          "  [nx <npts>] [out <file>] [statsout <file>] [name <name>]\n"
          "Fit datasets using a selected regression model (default: linear).\n"
          "Uncertainties and covariance matrix are printed to statsout.\n");
}

Analysis::RetType
Analysis_Regression::Setup(ArgList& al, AnalysisSetup& setup, int debugIn)
{

  //first thing, determine what kind of fit we are doing.
  std::string model_name = al.GetStringKey("model");
  if ( !model_name.empty() ) {
    fitModel_ = model_name;
  } else {
    fitModel_ = "linear";  
  }

  //optimiser parameters, with sane defaults.
  lm_tol_   = al.getKeyDouble("tol", 1e-12);
  lm_maxit_ = al.getKeyInt("maxit", 200);

  // create a (single) fit model: it can be re-parameterised
  // per dataset.
  model_ = ModelRegistry::Instance().Create(fitModel_);
  if (!model_) {
    mprinterr("Unknown model '%s'. Valid models are:\n", fitModel_.c_str());
    std::vector<std::string> all = ModelRegistry::Instance().List();
    for (std::vector<std::string>::const_iterator it = all.begin();
         it != all.end(); ++it)
      mprinterr("  %s\n", it->c_str());
    return Analysis::ERR;
  }
  if (model_->NumParams() <= 0 || model_->NumParams() >= 3) { 
    // sanity: so far, only models up to three parameters are implemented.
    mprinterr("regression: unexpected NumParams=%d for model '%s'\n", 
		    model_->NumParams(), fitModel_.c_str() );
    return Analysis::ERR;
  }

  ///////////////////below code copied from single-model version of the regression
  //with only few changes.
  nx_ = al.getKeyInt("nx", -1);
  if (nx_ > -1 && nx_ < 2) {
    mprinterr("Error: 'nx' must be greater than 1 if specified.\n");
    return Analysis::ERR;
  }
  DataFile* outfile = setup.DFL().AddDataFile(al.GetStringKey("out"), al);

  //for backward compatibility, the linear model has capital L.
  if( fitModel_ == "linear" ){
      statsout_ = setup.DFL().AddCpptrajFile(al.GetStringKey("statsout"),
                           "Linear regression stats", DataFileList::TEXT, true);
  } else {
      statsout_ = setup.DFL().AddCpptrajFile(al.GetStringKey("statsout"),
                           fitModel_ + " regression stats", DataFileList::TEXT, true);
  }
  if (statsout_ == 0) return Analysis::ERR;
  std::string setname = al.GetStringKey("name");

  // Select datasets from remaining args
  if (input_dsets_.AddSetsFromArgs( al.RemainingArgs(), setup.DSL() )) {
    mprinterr("Error: Could not add data sets.\n");
    return Analysis::ERR;
  }
  if (input_dsets_.empty()) {
    mprinterr("Error: No input data sets.\n");
    return Analysis::ERR;
  }

  // Setup output data sets
  int idx = 0;
  if ( input_dsets_.size() == 1 )
    idx = -1; // Only one input set, no need to refer to it by index
  // If setname is empty generate a default name
  
  if (setname.empty())
    setname = setup.DSL().GenerateDefaultName( "LR" );
  
  DataSet::DataType dtype = (nx_ > 1 ? DataSet::DOUBLE : DataSet::XYMESH);
  for ( Array1D::const_iterator DS = input_dsets_.begin();
                                DS != input_dsets_.end(); ++DS, idx++)
  {
    DataSet* dsout = setup.DSL().AddSet( dtype, MetaData(setname, idx) );
    if (dsout==0) return Analysis::ERR;
    dsout->SetLegend( "LR(" + (*DS)->Meta().Legend() + ")" );
    output_dsets_.push_back( (DataSet_1D*)dsout );

    //will now output automatically once filled. 
    if (outfile != 0) outfile->AddDataSet( dsout );
    
    //replacing the save of two datasets, slope and int(ercept) with 2*Num_params, for fit 
    //params and also errors of those params.
    const int M = model_->NumParams();
    std::vector<DataSet*> valueSets;
    std::vector<DataSet*> errorSets;
    valueSets.reserve(M);
    errorSets.reserve(M);
    
    //it seems ridiculous, but create an entire datset object for each scalar parameter
    //and another one for its ESE.
    for ( int p = 0; p < M; p++ ) {
      char pname[16];
      if ( fitModel_ == "linear" ) { 
	if ( p == 0 ) sprintf(pname, "intercept");
	else          sprintf(pname, "slope");
      } else {
        pname[0] = char( 'A' + p );
        pname[1] = '\0';	
      }

      //parameter Value dataset
      DataSet* vds =
          setup.DSL().AddSet(DataSet::DOUBLE,
                             MetaData(setname, pname, idx));
      valueSets.push_back(vds);
      
      //parameter Error dataset
      std::string errName = std::string(pname) + "_err";
      DataSet* eds =
          setup.DSL().AddSet(DataSet::DOUBLE,
                             MetaData(setname, errName.c_str(), idx));
      errorSets.push_back(eds);

    }

    // Store for Analyze()
    paramSets_.push_back(valueSets);
    paramErrSets_.push_back(errorSets);
    
  }

  mprintf("    REGRESSION: Calculating linear regression of %zu data sets.\n",
          input_dsets_.size());
  if (outfile != 0)
    mprintf("\tFit line output to %s\n", outfile->DataFilename().full());
  mprintf("\tFit statistics output to %s\n", statsout_->Filename().full());
  if (nx_ > 1)
    mprintf("\tUsing %i X values from input set min to max\n", nx_);
  else
    mprintf("\tUsing X values from input sets\n");

  return Analysis::OK;


}

Analysis::RetType Analysis_Regression::Analyze() {

  for (unsigned int iset=0; iset<input_dsets_.size(); ++iset) {
    const DataSet_1D* DS = input_dsets_[iset];

    if ((long int)DS->Size() < model_->NumParams()) {
      mprintf("regression WARNING: set '%s' has too few points (%zu).\n",
              DS->legend(), DS->Size());
      continue;
    }

    const int N = (int)DS->Size();
    std::vector<double> x(N), y(N);
    for (int i=0;i<N;i++) {
      x[i] = DS->Xcrd(i);
      y[i] = DS->Dval(i);
    }

    // Initial guess
    std::vector<double> params;
    model_->InitialGuess(x, y, params);

    /////////////Do the fit and collect params and uncertainties
    double rss=0.0;
    std::vector<double> cov, stderrv;
    const bool ok = LM_Optimize(*model_, x, y, params, rss, cov, stderrv, lm_maxit_, lm_tol_);
    (void)ok; // ok can be used for additional diagnostics if desired
    /////////////////////////////////////////////////////////////

    if (!statsout_->IsStream()) {
      //let the DataSet_1D class chunder out lots of statistics.
      statsout_->Printf("#Stats for %s\n", DS->legend());
    } 

    const std::vector<std::string> pnames = model_->ParamNames();
    double slope, intercept, correl, Fval; //outputs for legacy fit code.
    if ( fitModel_ == "linear" ) { 
      // Linear regression is (historically) farmed out to the DataSet_1D class.
      // We could just do it here, but for backwards compatibility the easiest way
      // is just to go on letting the legacy approach operate.      
      DS->LinearRegression( slope, intercept, correl, Fval, statsout_ );
    }
    else {
      // General (non-linear) models get a briefer output, delivered locally for now.
      if (statsout_ != 0) {
        statsout_->Printf("Model: %s\n", fitModel_.c_str());
        for (int k = 0; k < (int)params.size(); k++) {
          const double val = params[k];
          const double se  = (k < (int)stderrv.size() ? stderrv[k] : 0.0);
          statsout_->Printf("Param %s = %.12g ± %.6g\n", pnames[k].c_str(), val, se);
        }
        statsout_->Printf("RSS = %.12g\n", rss);
        statsout_->Printf("\n");
      }
    }

    // Build fitted curve
    DataSet_1D* out = output_dsets_[iset];

    mprintf("Building a plot of the fitted curve, with params: \n");
    for (int k = 0; k < (int)params.size(); k++) {
      mprintf("  %s : %.8f\n", pnames[k].c_str(), params[k] );
    }
    if ( fitModel_ == "linear" ) {
      mprintf("  ...legacy linear fit params (should match): %.8f %.8f\n", intercept, slope);
    }


    //
    if (nx_ < 2) {
      // We know the dataset is a DataSet_Mesh because Setup() used XYMESH
      DataSet_Mesh* mesh = dynamic_cast<DataSet_Mesh*>(out);
      mesh->Clear(); //if out dataset is a mesh
      // same X as input
      for (int i=0;i<N;i++) {
        const double yfit = model_->Evaluate(x[i], params);
        mesh->AddXY(x[i], yfit);
      }
    } else {
      // uniform grid min..max
      double xmin = x[0], xmax = x[0];
      for (int i=1;i<N;i++) { xmin = std::min(xmin, x[i]); xmax = std::max(xmax, x[i]); }
      const double step = (xmax - xmin) / (double)(nx_ - 1);
      double xv = xmin;
      for (int i=0;i<nx_;i++) {
        const double yv = model_->Evaluate(xv, params);
	out->Add( i, &yv );
        xv += step;
      }
      out->SetDim( Dimension::X, Dimension(xmin, step, "X"));
    }
    

    // attach fit parameters to datasets created in Setup()
    const std::vector<DataSet*>& psets = paramSets_[iset];
    const std::vector<DataSet*>& perrs = paramErrSets_[iset];
    FillParamSetsAndAttach(psets, params, perrs, stderrv, outfile_);

  
  }
  return Analysis::OK;
}


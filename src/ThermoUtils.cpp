#include "ThermoUtils.h"
#include <cmath>
#include <cstdio>
#include "CpptrajStdio.h"

namespace ThermoUtils {

//
//This is a utility function to calculate and print
//rigid-body thermochemical output, following MacQuarrie.
//
//See DataSet_Modes 
//and Analysis_Entropy 
//
//for more information about configurational entropy calculations,
//the information in this section is mainly just for 
//the relatively trivial rigid body terms.
//
int ComputeThermochemistry(CpptrajFile& outfile, const ThermoInput& in, int ilevel ) 
{
  if (!outfile.IsOpen()) {
     mprinterr("ThermoUtils: output file is not open.\n");
     return 1;
  }

  // Constants (copied from DataSet_Modes::Thermo)
  const double boltz = 1.380622e-23;     
  const double planck = 6.626196e-34;   
  const double jpcal = 4.18674e+00;      
  const double tomet = 1.0e-10;          
  const double hartre = 4.35981e-18;     
  const double pstd = 1.01325e+05;       
  const double pipi = Constants::PI * Constants::PI;
  const double e = std::exp(1.0);
  const double tocal = 1.0 / jpcal;
  const double tokcal = tocal / 1000.0;

  double temp     = in.temperature;
  double pressure = pstd * in.pressure_atm;

 //     print the temperature and pressure.
  outfile.Printf("\n                    *******************\n");
  outfile.Printf(  "                    - Thermochemistry -\n");
  outfile.Printf(  "                    *******************\n\n");
  outfile.Printf("\n temperature %9.3f kelvin\n pressure %9.5f atm\n", temp, in.pressure_atm);

  // Molecular mass
  double weight = 0.0;
  for (double m : *(in.masses_amu)) weight += m;
  outfile.Printf(" molecular mass (principal isotopes) %11.5f amu\n", weight);

  double weight_kg = weight * Constants::AMU_TO_KG;
  double rt = Constants::GASK_J * temp;

  // -------- Translational contributions ----------
  double dum1 = boltz * temp;
  double dum2 = std::pow(Constants::TWOPI, 1.5);

  double arg = std::pow(dum1, 1.5) / planck;
  arg = (arg / pressure) * (dum1 / planck);
  arg = arg * dum2 * (weight_kg / planck);
  arg = arg * std::sqrt(weight_kg) * std::exp(2.5);

  double stran = Constants::GASK_J * std::log(arg);
  double etran = 1.5 * rt;
  double ctran = 1.5 * Constants::GASK_J;

  //handle trivial case of single atom
  if ( in.avg_coords->size() <= 3 ) {
    outfile.Printf("\n internal energy: %10.3f joule/mol %10.3f kcal/mol\n",
                       etran, etran * tokcal);
    outfile.Printf(" entropy:         %10.3f joule/k-mol %10.3f cal/k-mol\n",
                       stran, stran * tocal);
    outfile.Printf(" heat capacity:   %10.3f joule/k-mol %10.3f cal/k-mol\n",
                       ctran, ctran * tocal);
    return 0;
  }

  // Build frame
  Frame AVG;
  AVG.SetupFrameXM( *(in.avg_coords), *(in.masses_amu) );

  // Inertia tensor
  Matrix_3x3 Inertia;
  AVG.CalculateInertia(AtomMask(0, AVG.Natom()), Inertia);

  Vec3 pmom;
  Inertia.Diagonalize_Sort(pmom);

  double tmp = pmom[0];
  pmom[0] = pmom[2];
  pmom[2] = tmp;

  outfile.Printf("\n principal moments of inertia (nuclei only) in amu-A**2:\n");
  outfile.Printf(" %12.2f%12.2f%12.2f\n", pmom[0], pmom[1], pmom[2]);

  //This seems like a horrible hacky way to estimate if an atom is linear
  //What about CO2?
  bool linear = false;
  double sn = 1.0;
  if (AVG.Natom() <= 2) {
    linear = true;
    if (AVG.Mass(0) == AVG.Mass(1)) sn = 2.0;
  }
  outfile.Printf("\n rotational symmetry number %3.0f\n", sn);

  double con = planck / (boltz * 8.0 * pipi);
  con = (con / Constants::AMU_TO_KG) * (planck / (tomet*tomet));
  double r1, r2, r3;
  if (linear) {
    r3 = con / pmom[2];
    if( r3 < 0.2 ) {
      outfile.Printf("\n Warning-- assumption of classical behavior for rotation\n");
      outfile.Printf(  "           may cause significant error\n");
    }
    outfile.Printf("\n rotational temperature (kelvin) %12.5f\n", r3);
  } else {
    r1 = con / pmom[0];
    r2 = con / pmom[1];
    r3 = con / pmom[2];
    if ( r1 < 0.2 ) {
      outfile.Printf("\n Warning-- assumption of classical behavior for rotation\n");
      outfile.Printf(  "           may cause significant error\n");
    }
    outfile.Printf("\n rotational temperatures (kelvin) %12.5f%12.5f%12.5f\n",r1, r2, r3);
  }

  double erot, crot, srot;
  if (linear) {
    erot = rt;
    crot = Constants::GASK_J;
    arg = (temp / r3) * (e / sn);
    srot = Constants::GASK_J * std::log(arg);
  } else {
    erot = 1.5 * rt;
    crot = 1.5 * Constants::GASK_J;
    arg = std::sqrt(Constants::PI * e * e * e) / sn;
    double dum = (temp/r1)*(temp/r2)*(temp/r3);
    arg = arg * std::sqrt(dum);
    srot = Constants::GASK_J * std::log(arg);
  }

  // Vibrations
  int ndof = in.nmodes;
  int iff;

  if (ilevel != 0)
    iff = 0;
  else if (linear)
    iff = 5;
  else
    iff = 6;

  if ( iff > 0 && ndof > 0 )
    outfile.Printf("The first %i frequencies will be skipped.\n", iff);

  // Can only output per-mode vibrational terms if we have a normal-modes
  // or pseudo-normal (PCA) output of a spectrum.
  //
  // Other methods can estimate a configurational entropy and still output
  // something comparable to the total of the harmonic-oscillator entropies
  // without an individual mode decomposition.
  double evib = 0.0, cvib = 0.0, svib = 0.0;
  //siwtch on ndof to see if we have modewise info.
  if ( ndof > 0 ) {
    std::vector<double> vtemp(ndof);
    double con2 = planck / boltz;
    double ezpe = 0.0;
    for (int i = 0; i < ndof; i++) {
      vtemp[i] = in.freqs_cm1[i] * con2 * 3.0e10;
      ezpe += in.freqs_cm1[i] * 3.0e10;
    }

    ezpe = 0.5 * planck * ezpe;

    outfile.Printf("\n zero point vibrational energy %12.1f (joules/mol)\n"
                   "                                  %12.5f (kcal/mol)\n"
                   "                                  %12.7f (hartree/particle)\n",
                   ezpe*Constants::NA, ezpe*tokcal*Constants::NA, ezpe/hartre);

    int lofreq = 0;
    for (double vt : vtemp) if (vt < 900.0) lofreq++;
    if (lofreq) {
      outfile.Printf("\n Warning-- %3i vibrations have low frequencies and may represent hindered \n", lofreq);
      outfile.Printf(  "         internal rotations.  The contributions printed below assume that these \n");
      outfile.Printf(  "         really are vibrations.\n");
    }

    for (int i = 0; i < ndof; i++) {
        double tovt = vtemp[i] / temp;
        double etovt = std::exp(tovt);
        double em1 = etovt - 1.0;

        double econt = tovt * (0.5 + 1.0/em1);
        double ccont = etovt * std::pow(tovt/em1, 2.0);

        double argd = 1.0 - 1.0/etovt;
        double scont = 0.;
       	if ( argd > 1e-7 ) {
	  scont = tovt/em1 - std::log(argd);
	} else {
	  outfile.Printf(" warning: setting vibrational entropy to zero for mode %i with vtemp = %f\n",
               i+1, vtemp[i]);
	}

        evib += econt;
        cvib += ccont;
        svib += scont;
    }

    evib *= rt;
    cvib *= Constants::GASK_J;
    svib *= Constants::GASK_J;
  }

  // Convert everything
  double Etran = etran * tokcal;
  double Ctran = ctran * tocal;
  double Stran = stran * tocal;

  double Erot = erot * tokcal;
  double Crot = crot * tocal;
  double Srot = srot * tocal;

  double Etot = Etran + Erot;
  double Ctot = Ctran + Crot;
  double Stot = Stran + Srot;

  double Evib = 0., Cvib = 0., Svib = 0.;
  if ( ndof > 0 ){
    Evib = evib * tokcal;
    Cvib = cvib * tocal;
    Svib = svib * tocal;

    Etot += Evib;
    Ctot += Cvib;
    Stot += Svib;
  }

  outfile.Printf("\n\n           freq.         E                  Cv                 S\n");
  outfile.Printf(    "          cm**-1      kcal/mol        cal/mol-kelvin    cal/mol-kelvin\n");
  outfile.Printf(    "--------------------------------------------------------------------------------\n");
  outfile.Printf(    " Total              %11.3f        %11.3f        %11.3f\n", Etot,  Ctot,  Stot);
  outfile.Printf(    " translational      %11.3f        %11.3f        %11.3f\n", Etran, Ctran, Stran);
  outfile.Printf(    " rotational         %11.3f        %11.3f        %11.3f\n", Erot,  Crot,  Srot);
  if ( ndof > 0 ) {
    outfile.Printf(  " vibrational        %11.3f        %11.3f        %11.3f\n", Evib,  Cvib,  Svib);
  }

  return 0;
}

} // namespace ThermoUtils


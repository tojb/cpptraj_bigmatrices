
#ifndef THERMO_UTILS_H
#define THERMO_UTILS_H

#include <vector>
#include "CpptrajFile.h"
#include "Vec3.h"
#include "Matrix_3x3.h"
#include "Frame.h"
#include "Constants.h"

// This utility prints the thermochemistry output 
// which was previously handled as a member function of DataSet_Modes::Thermo().
// 
// This was unpacked to a standalone module by JTB because I want to reuse it for
// Analysis_Entropy .
//

namespace ThermoUtils {

struct ThermoInput {
    const std::vector<double> *masses_amu;  // amu
    const std::vector<double> *avg_coords;  // Angstrom, 3N entries
    double                    *freqs_cm1;   // vibrational frequencies, ascending
    int nmodes;                       // number of vibrational modes

    double temperature;               // K
    double pressure_atm;              // atm
};

int ComputeThermochemistry(
    CpptrajFile& out,
    const ThermoInput& in,
    int ilevel    // as in DataSet_Modes::Thermo()
);

} // namespace ThermoUtils

#endif


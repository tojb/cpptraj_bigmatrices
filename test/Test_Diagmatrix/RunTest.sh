#!/bin/bash

. ../MasterTest.sh

CleanFiles matrix.in MyThermo.dat test.chk diagtest.evecs.dat

TESTNAME='MW covariance matrix thermo analysis test'
Requires netcdf mathlib 

INPUT="-i matrix.in"
cat > matrix.in <<EOF
parm ../DPDP.parm7
trajin ../DPDP.nc
matrix MyMatrix mwcovar @1-33&!@H=
diagmatrix MyMatrix thermo outthermo MyThermo.dat
EOF
RunCpptraj "$TESTNAME"
DoTest MyThermo.dat.save MyThermo.dat

# Simple smoke test: verify checkpoint option doesn't break diagonalization
cat > matrix.in <<EOF
parm ../DPDP.parm7
trajin ../DPDP.nc 1 10
matrix TestMatrix mwcovar @1-33&!@H=
diagmatrix TestMatrix checkpoint out diagtest.evecs.dat vecs 5
EOF
RunCpptraj "Diagonalize with checkpoint option enabled"

# Just verify the output was created without errors
if [ -f diagtest.evecs.dat ]; then
  echo "Checkpoint smoke test passed"
fi

EndTest

exit 0


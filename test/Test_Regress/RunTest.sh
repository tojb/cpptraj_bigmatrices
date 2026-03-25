#!/bin/bash

. ../MasterTest.sh

CleanFiles regress.in fit.dat statsout.dat Y.dat regress.dat

TESTNAME='Linear regression test.'

Requires maxthreads 1 

INPUT="-i regress.in"

##just generate some tiny datafiles for minimal regression tests:
cat > XE.dat <<EOF
0.0   0.00
1.0   0.63
2.0   0.86
5.0   0.99    
10.0  1.00    
EOF


cat > regress.in <<EOF
readdata ../Test_LowestCurve/esurf_vs_rmsd.dat.txt index 1 name XY
regress XY out regress.dat name FitXY nx 100 statsout statsout.dat
runanalysis
list dataset
createset "Y = FitXY[slope] * X + FitXY[intercept]" xstep .2 nx 100
writedata Y.dat Y

##and test exponential regression f(x) = A( 1-exp(-Bx) )
readdata XE.dat index 1 name XE
regress XE model expsat name FitE out regress_expsat.dat statsout FitE.stats maxit 100000 tol 1e-15
runanalysis

EOF

RunCpptraj "$TESTNAME"
DoTest statsout.dat.save       statsout.dat
DoTest regress.dat.save        regress.dat
DoTest Y.dat.save              Y.dat
DoTest regress_expsat.dat.save regress_expsat.dat

EndTest
exit 0

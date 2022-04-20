#!/bin/bash
prefix=abf_md
in_file=knk_in_file
echo $PWD
SUBMISSION=$PWD
export REAC_TMPDIR=$PWD/Reaction/${prefix}
mkdir -p ${REAC_TMPDIR}
cp -rp ${SUBMISSION}/*.py         ${REAC_TMPDIR}
cp -rp ${SUBMISSION}/pot.fs       ${REAC_TMPDIR}
cp -rp ${SUBMISSION}/${in_file}   ${REAC_TMPDIR}
SCRIPTDIR=$SUBMISSION
cp -rp $SCRIPTDIR/lib   ${REAC_TMPDIR}
#cp -rp $SCRIPTDIR/_knots     ${REAC_TMPDIR}
cp -rp $SCRIPTDIR/struc     ${REAC_TMPDIR}

outputfile=$SUBMISSION/abf_ae.out
ulimit -c unlimited


cd  ${REAC_TMPDIR}
for t in 300
do
python bABF_ae_cv.py -i ${in_file} -t $t |tee ${outputfile}.${t}
done






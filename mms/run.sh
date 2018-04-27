#!/bin/bash

NPROCS=$1

# Setup
WORKDIR=`pwd`
PELECBIN=${WORKDIR}/PeleC3d.gnu.MPI.ex
CASEDIR=${WORKDIR}/'cases'
NCELLS=(8 16 32 64)
CFLS=(0.1 0.2 0.3 0.7 0.9)

rm -rf ${CASEDIR}
mkdir -p ${CASEDIR}
cd ${CASEDIR}
for NCELL in "${NCELLS[@]}"
do
    for CFL in "${CFLS[@]}"
    do
        NCELL=${NCELLS[i]}
        CFL=${CFLS[i]}
        DT=${DTS[i]}
        DIR=${NCELL}cells_${CFL}cfl
        rm -rf ${DIR}
        mkdir -p ${DIR}
        cd ${DIR}
        cp ${WORKDIR}/inputs_3d ${WORKDIR}/probin .
        mpirun -np ${NPROCS} ${PELECBIN} inputs_3d pelec.cfl=${CFL} amr.n_cell=${NCELL} ${NCELL} ${NCELL} > out 2>&1;
        cd ${CASEDIR}
    done
done


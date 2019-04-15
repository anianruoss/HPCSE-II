module load new
module load gcc/6.3.0
module load intel/2018.1
module load impi/2018.1.163

export UPCXX_GASNET_CONDUIT=smp
export UPCXX_THREADMODE=seq
export UPCXX_CODEMODE=O3
export KORALI_CONDUIT=upcxx

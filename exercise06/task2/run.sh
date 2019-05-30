bsub -R fullnode -n 24 "mpirun -n 24 ./jacobi"
bsub -R fullnode -n 48 "mpirun -n 48 ./jacobi"
bsub -R fullnode -n 96 "mpirun -n 96 ./jacobi"

bsub -R fullnode -n 24 "unset LSB_AFFINITY_HOSTFILE; OMP_NUM_THREADS=6 mpirun -n 4 --map-by node:PE=6 --report-bindings ./jacobi_opt"
bsub -R fullnode -n 48 "unset LSB_AFFINITY_HOSTFILE; OMP_NUM_THREADS=6 mpirun -n 4 --map-by node:PE=6 --report-bindings ./jacobi_opt"
bsub -R fullnode -n 96 "unset LSB_AFFINITY_HOSTFILE; OMP_NUM_THREADS=6 mpirun -n 4 --map-by node:PE=6 --report-bindings ./jacobi_opt"

bsub -R fullnode -n 24 "unset LSB_AFFINITY_HOSTFILE; OMP_NUM_THREADS=12 mpirun -n 2 --map-by node:PE=12 --report-bindings ./jacobi_opt"
bsub -R fullnode -n 48 "unset LSB_AFFINITY_HOSTFILE; OMP_NUM_THREADS=12 mpirun -n 4 --map-by node:PE=12 --report-bindings ./jacobi_opt"
bsub -R fullnode -n 96 "unset LSB_AFFINITY_HOSTFILE; OMP_NUM_THREADS=12 mpirun -n 8 --map-by node:PE=12 --report-bindings ./jacobi_opt"


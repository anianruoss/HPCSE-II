bsub -R fullnode -n 24 "unset LSB_AFINITY_HOSTFILE; mpirun -n 24 ./jacobi"

bsub -R fullnode -n 24 "unset LSB_AFINITY_HOSTFILE ; OMP_NUM_THREADS=12 mpirun -n 2 --map-by node:PE=12 ./jacobi_opt"
bsub -R fullnode -n 48 "unset LSB_AFINITY_HOSTFILE ; OMP_NUM_THREADS=12 mpirun -n 4 --map-by node:PE=12 ./jacobi_opt"
bsub -R fullnode -n 96 "unset LSB_AFINITY_HOSTFILE ; OMP_NUM_THREADS=12 mpirun -n 8 --map-by node:PE=12 ./jacobi_opt"

bsub -R fullnode -n 24 "unset LSB_AFINITY_HOSTFILE ; OMP_NUM_THREADS=6 mpirun -n 4 --map-by node:PE=6 ./jacobi_opt"
bsub -R fullnode -n 48 "unset LSB_AFINITY_HOSTFILE ; OMP_NUM_THREADS=6 mpirun -n 4 --map-by node:PE=6 ./jacobi_opt"
bsub -R fullnode -n 96 "unset LSB_AFINITY_HOSTFILE ; OMP_NUM_THREADS=6 mpirun -n 4 --map-by node:PE=6 ./jacobi_opt"

bsub -R fullnode -n 24 "unset LSB_AFINITY_HOSTFILE ; OMP_NUM_THREADS=6 mpirun -n 4 --map-by node:PE=6 ./jacobi_opt"
bsub -R fullnode -n 48 "unset LSB_AFINITY_HOSTFILE ; OMP_NUM_THREADS=12 mpirun -n 4 --map-by node:PE=12 ./jacobi_opt"
bsub -R fullnode -n 96 "unset LSB_AFINITY_HOSTFILE ; OMP_NUM_THREADS=24 mpirun -n 4 --map-by node:PE=24 ./jacobi_opt"

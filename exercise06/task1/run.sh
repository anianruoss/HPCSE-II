bsub -R fullnode -n 24 "./heat2d_cpu"
bsub -R fullnode -n 24 "mpirun -n 1 ./heat2d_mpi"
bsub -R fullnode -n 24 "mpirun -n 24 ./heat2d_mpi"
bsub -R fullnode -n 48 "mpirun -n 48 ./heat2d_mpi"
bsub -R fullnode -n 96 "mpirun -n 96 ./heat2d_mpi"


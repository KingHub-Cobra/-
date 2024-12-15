compile:
	module avail && module load SpectrumMPI/10.1.0 && mpicc main_openMP_MPI.c -lm -fopenmp
submit_160_180_1_1:
	bsub -n 1 -q normal -W "3:00" -o 160x180_1_1.out -eo 160x180_1_1.err -m "polus-c2-ib" -R "affinity[thread(1,same=core)*1]" OMP_NUM_THREADS=1 mpiexec -n 1 ./a.out
submit_160_180_1_4:
	bsub -n 4 -q normal -W "3:00" -o 160x180_1_4.out -eo 160x180_1_4.err -m "polus-c2-ib polus-c3-ib" -R "span[ptile=2] affinity[thread(1,same=core)*4]" OMP_NUM_THREADS=1 mpiexec -n 4 ./a.out
submit_160_180_2_4:
	bsub -n 4 -q normal -W "3:00" -o 160x180_2_4.out -eo 160x180_2_4.err -m "polus-c2-ib polus-c3-ib" -R "span[ptile=2] affinity[thread(2,same=core)*4]" OMP_NUM_THREADS=2 mpiexec -n 4 ./a.out
submit_160_180_4_4:
	bsub -n 4 -q normal -W "3:00" -o 160x180_4_4.out -eo 160x180_4_4.err -m "polus-c2-ib polus-c3-ib" -R "span[ptile=2] affinity[thread(4,same=core)*4]" OMP_NUM_THREADS=4 mpiexec -n 4 ./a.out
submit_160_180_8_4:
	bsub -n 4 -q normal -W "3:00" -o 160x180_8_4.out -eo 160x180_8_4.err -m "polus-c2-ib polus-c3-ib" -R "span[ptile=2] affinity[thread(8,same=core)*4]" OMP_NUM_THREADS=8 mpiexec -n 4 ./a.out



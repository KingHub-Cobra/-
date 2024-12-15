compile:
	module avail && module load SpectrumMPI/10.1.0 && mpicc main_openMP_MPI.c -lm -fopenmp
submit_80_90_1_1:
	bsub -n 1 -q normal -W "1:00" -o 80x90_1_1.out -eo 80x90_1_1.err -m "polus-c3-ib" -R "affinity[thread(1,same=core)*1]" OMP_NUM_THREADS=1 mpiexec -n 1 ./a.out
submit_80_90_2_1:
	bsub -n 1 -o 80x90_2_1.out -eo 80x90_2_1.err -m "polus-c3-ib" OMP_NUM_THREADS=2 mpiexec -n 1 ./a.out
submit_80_90_4_1:
	bsub -n 1 -o 80x90_4_1.out -eo 80x90_4_1.err -m "polus-c3-ib" OMP_NUM_THREADS=4 mpiexec -n 1 ./a.out
submit_80_90_1_2:
	bsub -n 2 -o 80x90_1_2.out -eo 80x90_1_2.err -m "polus-c3-ib" -R "affinity[thread(1,same=core)*2]" OMP_NUM_THREADS=1 mpiexec -n 2 ./a.out
submit_80_90_2_2:
	bsub -n 2 -o 80x90_2_2.out -eo 80x90_2_2.err -m "polus-c2-ib polus-c3-ib" -R "span[ptile=1] affinity[thread(2,same=core)*2]"  OMP_NUM_THREADS=2 mpiexec -n 2 ./a.out
submit_80_90_4_2:
	bsub -n 2 -o 80x90_4_2.out -eo 80x90_4_2.err -m "polus-c3-ib" -R "span[ptile=2] affinity[thread(4,same=core)*2]" OMP_NUM_THREADS=4 mpiexec -n 2 ./a.out
submit_80_90_8_2:
	bsub -n 2 -o 80x90_8_2_a.out -eo 80x90_8_2_a.err -m "polus-c2-ib polus-c3-ib" -R "span[ptile=1] affinity[thread(8,same=core)*2]"  OMP_NUM_THREADS=8 mpiexec -n 2 ./a.out
submit_80_90_1_4:
	bsub -n 4 -o 80x90_1_4.out -eo 80x90_1_4.err -m "polus-c4-ib polus-c3-ib" -R "span[ptile=2] affinity[core(4)]" OMP_NUM_THREADS=1 mpiexec -n 4 ./a.out
submit_80_90_2_4:
	bsub -n 4 -o 80x90_2_4.out -eo 80x90_2_4.err -m "polus-c4-ib polus-c3-ib" -R "span[ptile=2] affinity[core(4)]" OMP_NUM_THREADS=2 mpiexec -n 4 ./a.out
submit_80_90_4_4:
	bsub -n 4 -o 80x90_4_4.out -eo 80x90_4_4.err -m "polus-c4-ib polus-c3-ib" -R "span[ptile=2] affinity[core(4)]" OMP_NUM_THREADS=4 mpiexec -n 4 ./a.out



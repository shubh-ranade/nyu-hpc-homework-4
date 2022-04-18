mmult : mmult.cu
	nvcc -std=c++11 mmult.cu -o mmult

jacobi : driver.cu jacobi.cu jacobi.cuh
	nvcc -std=c++11 driver.cu jacobi.cu -o jacobi2D-cuda

clean:
	rm -f *.out
	rm jacobi2D-cuda
	rm mmult

#include <iostream>
#include "jacobi.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <string>

void swap(double **r, double **s)
{
    double *pSwap = *r;
    *r = *s;
    *s = pSwap;
}

int main(int argc, char * argv[]) 
{
    int N;
    if(argc < 2) N = 100; // default if no value given
    else N = std::stoi(argv[1]);

    int max_iter;
    if(argc < 3) max_iter = 1000; // default
    else max_iter = std::stoi(argv[2]);

    int dimensions[2] = {N, N}, // The dimensions of the grid
        nthreads = N / 10 + 1, // Number of CUDA threads per CUDA block dimension. 
        // nthreads = N,
        u_mem_required = dimensions[0] * dimensions[1] * sizeof(double);
    

    double * start_u, * f, * u, * u_new, * f_device;
    const dim3 blockSize( nthreads , nthreads),
               gridSize( (dimensions[0] + nthreads - 1) / nthreads, (dimensions[1] + nthreads - 1) / nthreads);
    // const dim3 blockSize( 10 , 10), // The size of CUDA block of threads.
            //    gridSize( dimensions[0] / 10, dimensions[1] / 10 );

    std::cout << "Initializing u to 0 and f to 1" << std::endl;
    
    // Initialize u and f
    start_u = new double[dimensions[0] * dimensions[1]];
    f = new double[dimensions[0] * dimensions[1]];
    for(int i = 0; i < dimensions[0]; i++) {
        int offset = i * dimensions[1];
        for(int j = 0; j < dimensions[1]; j++) {
            start_u[offset + j] = 0;
            f[offset + j] = 1;
        }
    }

    std::cout << "Initial residual = " << calculate_residual(start_u, f, dimensions) << std::endl;

    // Need to copy start_u from host to CUDA device.
    std::cout << "Copying to Device" << std::endl;
    try 
    {
        copyToDevice(start_u, f, dimensions, &u, &u_new, &f_device);
    }
    catch( ... )
    {
        std::cout << "Exception happened while copying to device" << std::endl;
    }

    std::cout << "Perform Jacobi iterations" << std::endl;
    for( int i = 0; i < max_iter; i++)
    {
        // Call CUDA device kernel to do a Jacobi iteration. 
        doJacobiIteration<<< gridSize, blockSize >>>(dimensions[0], dimensions[1], u, u_new, f_device);
        cudaDeviceSynchronize();
        if(cudaGetLastError() != cudaSuccess)
        {
            std::cout << "Error Launching Kernel" << std::endl;
            return 1;
        }
        // swap(&u, &u_new);
        cudaMemcpyAsync( u, u_new, u_mem_required, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();        
    }

    // Get the result from the CUDA device.
    std::cout << "Copying result back to start_u" << std::endl;
    if(cudaMemcpy( start_u, u, u_mem_required, cudaMemcpyDeviceToHost ) != cudaSuccess) 
    {
        std::cout << "There was a problem retrieving the result from the device" << std::endl;
        return 1;    
    }

    // std::cout << "Final u\n";
    // for(int j = 0; j < N; j++)
    //     std::cout << start_u[j] << ' ';
    // std::cout << '\n';
    
    // Final residual
    std::cout << "Final Jacobi residual = " << calculate_residual(start_u, f, dimensions) << std::endl;

    // Clean up memory.
    cudaFree(u);
    cudaFree(u_new);
    delete [] start_u;
    delete [] f;

    return 0;
}

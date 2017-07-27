#include <stdlib.h>


// CUDA Kernel
// int n - vector length, maybe not friendly multiple of blockDim.x
__global__ void vecAdd(float *in1, float *in2, float *out, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
        out[index] = in1[index] + in2[index];
}



#define N (1024 * 1024 + 10)    // vector length == number of workers
#define TH_PER_BLK 512          // threads per thread block



// Init N elements of array of floats
void random_floats(float *a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = rand();
    }
}



int main(void) {

    float *in1, *in2, *out;
    float *d_in1, *d_in2, *d_out;
    int size = N * sizeof(float);

    // Alloc/Init Host array/vector of in1, in2, out. App specific, Just one example
	in1 = (float *)malloc(size); random_floats(in1, N);
    in2 = (float *)malloc(size); random_floats(in2, N);
    out = (float *)malloc(size);

    // Alloc global memory on device for copies of Host array/vector: in1, in2, out
    cudaMalloc((void **)&d_in1, size);
    cudaMalloc((void **)&d_in2, size);
    cudaMalloc((void **)&d_out, size);

    // Copy input vectors from Host to Device
    cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);

    // Launch Cuda Kernel on GPU, Parallel Execute, handle N isn't multiple blks
    vecAdd<<<(N+TH_PER_BLK-1)/TH_PER_BLK, TH_PER_BLK>>>(d_in1, d_in2, d_out, N);

    // Copy result from GPU global memory to Host
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    // Note: cudaMemcpy (sync) after kernel launch (async) acts as a barrier, otherwise need
    // cudaDeviceSynchronize()

    // Consuming vector of out is ignored here

    // free up memory
    cudaFree(d_in1); cudaFree(d_in2); cudaFree(d_out);
    free(in1); free(in2); free(out);

    return 0;
}

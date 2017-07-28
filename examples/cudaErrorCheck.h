#ifndef _CUDA_ERR_CHK_

#define _CUDA_ERR_CHK_

#include <stdio.h>

// Define this to turn on error checking
#define CUDA_ERR_CHK 1

#define CudaApiCall( err ) __cudaCheckApi( err, __FILE__, __LINE__ )
#define CudaChkKern()    __cudaCheckKern( __FILE__, __LINE__ )

// Check Errors from API call
inline void __cudaCheckApi( cudaError err, const char *file, const int line )
{
#if CUDA_ERR_CHK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

// Check Errors from Kernel call
inline void __cudaCheckKern( const char *file, const int line )
{
#if CUDA_ERR_CHK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

#endif  // _CUDA_ERR_CHK_

#include "cuda_runtime.h"
#include <stdio.h>

__global__ void simpleKernel( int *a )
{

/* calculate my global index in the array */

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

/* assign value to the output array */
/* change code here to change the output */

	a[idx] = 7;
//	a[idx] = blockIdx.x;
//	a[idx] = threadIdx.x;
} /* end simpleKernel */ 


int main()
{
	int dimx = 16;
	int numbytes = dimx * sizeof( int );

/* declare the device and host pointers */

	int *d_a = 0, *h_a = 0; // device and host pointers

/* allocate the memory on host and device */

	h_a = (int *) malloc( numbytes );
	cudaMalloc( (void **) &d_a, numbytes );

	if( 0 == h_a || 0 == d_a )
	{
		printf("Couldn't allocate memory!\n");
		return 911;
	} /* end if */

/* initialize GPU memory to 0 */

	cudaMemset( d_a, 0, numbytes );

/* setup GPU grid and block */

	dim3 mygrid, myblock;

	myblock.x = 4;
	mygrid.x = dimx / myblock.x;

/* launch the kernel */

	simpleKernel<<< mygrid, myblock >>>( d_a );

/* copy result back to GPU */

	cudaMemcpy( h_a, d_a, numbytes, cudaMemcpyDeviceToHost );

/* check GPU and CPU data to ensure they are equal */

	for( int i = 0; i < dimx; i++ )
	{
		printf("%d ", h_a[i] );
	} /* end for */
	printf("\n");

/* free the memory and cleanup */

	free( h_a );
	cudaFree( d_a );

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 911;
    }

    return 0;
}
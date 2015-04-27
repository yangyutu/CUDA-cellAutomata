#include <stdio.h>
#include "para.h"



// the kernel to udpate the state of each cell based on its neighbors for certain number of steps
__global__ void update(int *out, int *in, int dim){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y*blockDim.x*gridDim.x;
	int sum=0;
	for(int i=-1;i<2;i++){
		for(int j=-1;j<2;j++){
			int xtemp=(x+i+dim)%dim;
			int ytemp=(y+j+dim)%dim;
			int offsettemp=xtemp + ytemp*blockDim.x*gridDim.x;
			sum=sum+in[offsettemp];
		}
	}
	sum=sum-in[offset];
	if (in[offset] == 1) {
		if(sum == 2 || sum ==3) out[offset] = 1;
		else out[offset] = 0;
	}else{
	if( sum == 3) out[offset] = 1;
	}

}
	
int main(int argc, char *argv[]){

	int dim=100;
	int nStep=1000;
	int frequency=100;
	int size=dim*dim;
	int step;
	DataBlock data;
	data.bitmap=(int *)malloc(size*sizeof(int));
	int bitmapSize=size*sizeof(int);

	HANDLE_ERROR(cudaMalloc( (void **)&(data.dev_in),bitmapSize));
	HANDLE_ERROR(cudaMalloc( (void **)&(data.dev_out),bitmapSize));
//	HANDLE_ERROR(cudaMalloc( (void **)&(data.dev_in),size));
	
//	HANDLE_ERROR(cudaBindTexture (NULL, texIn, data.dev_in, bitmapSize));
//	HANDLE_ERROR(cudaBindTexture (NULL, texOut, data.dev_out, bitmapSize));



	HANDLE_ERROR(cudaMemcpy(data.dev_in, data.bitmap, bitmapSize, cudaMemcpyHostToDevice));

	dim3 dimgrid(dim/16,dim/16);
	dim3 dimblock(16,16);

	for(step = 0; step<nStep; step++ ){
	
	update<<<dimgrid,dimblock>>>(data.dev_in,data.dev_out,dim);
	
	swap(data.dev_in,data.dev_out);

	if(step % frequency == 0){
	HANDLE_ERROR(cudaMemcpy(data.outbitmap,data.dev_out,bitmapSize,cudaMemcpyDeviceToHost));

      printf ( "\nIteration %d: final grid:\n", step );
      for ( int j = 0; j < size; j++ ) {
        if ( j % dim == 0 ) {printf( "\n" );}
        printf ( "%d  ", data.outbitmap[j] );
      }
      printf( "\n" );
	}
    	}

	HANDLE_ERROR(cudaFree(data.dev_in));
	HANDLE_ERROR(cudaFree(data.dev_out));

}

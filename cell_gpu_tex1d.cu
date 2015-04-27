#include <stdio.h>
#include "para.h"

texture<int> texIn;
texture<int> texOut;

// the kernel to udpate the state of each cell based on its neighbors for certain number of steps
__global__ void update_tex1d(int *out, int flag,int dim){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y*blockDim.x*gridDim.x;
	int sum=0;
	int origin;
	
	for(int i=-1;i<2;i++){
		for(int j=-1;j<2;j++){
			int xtemp=(x+i+dim)%dim;
			int ytemp=(y+j+dim)%dim;
			int offsettemp=xtemp + ytemp*blockDim.x*gridDim.x;
			if (flag == 1)			
				sum=sum+tex1Dfetch(texIn,offsettemp);
			else
				sum=sum+tex1Dfetch(texOut,offsettemp);
		}
	}
			if (flag == 1)			
				origin=tex1Dfetch(texIn,offset);
			else
				origin=tex1Dfetch(texOut,offset);
	
	sum=sum-origin;
	if (origin == 1) {
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
	int flag;
	HANDLE_ERROR(cudaMalloc( (void **)&(data.dev_in),bitmapSize));
	HANDLE_ERROR(cudaMalloc( (void **)&(data.dev_out),bitmapSize));
//	HANDLE_ERROR(cudaMalloc( (void **)&(data.dev_in),size));
	
	HANDLE_ERROR(cudaBindTexture (NULL, texIn, data.dev_in, bitmapSize));
	HANDLE_ERROR(cudaBindTexture (NULL, texOut, data.dev_out, bitmapSize));



	HANDLE_ERROR(cudaMemcpy(data.dev_in, data.bitmap, bitmapSize, cudaMemcpyHostToDevice));

	dim3 dimgrid(dim/16,dim/16);
	dim3 dimblock(16,16);

	flag = 1;
	for(step = 0; step<nStep; step++ ){
	int *in,*out;

	if( flag == 1 ) {out = data.dev_out;flag = 0;}
	else {out=data.dev_in; flag = 1;}

	
	update_tex1d<<<dimgrid,dimblock>>>(out,flag,dim);
	
//	swap(data.dev_in,data.dev_out);

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

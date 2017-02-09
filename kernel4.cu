#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define PIXELSIZE 4


double* ClusterCore;
unsigned int*  ClusterPixelCount;
double* ClusterColorSum;

__constant__ double DevClusterCore[4*32];

__global__ void PixelToCluster(unsigned char* Image, int ClusterCount, int Width, int Height, int* LastIter)
{
	int ElementsInStr=PIXELSIZE*Width;
	int Distance=0;
	int SelectedCluster=0;
	int a,b,c;  

	int i,j;

	for(int offset=4*(blockIdx.x*blockDim.x+threadIdx.x); offset<Height*ElementsInStr; offset+=4*(gridDim.x*blockDim.x))
	{
	   i=offset/ElementsInStr;
	   j=offset%ElementsInStr;

	   a=(DevClusterCore[0]-Image[i*ElementsInStr+j])*(DevClusterCore[0]-Image[i*ElementsInStr+j]);
	   b=(DevClusterCore[1]-Image[i*ElementsInStr+j+1])*(DevClusterCore[1]-Image[i*ElementsInStr+j+1]);
	   c=(DevClusterCore[2]-Image[i*ElementsInStr+j+2])*(DevClusterCore[2]-Image[i*ElementsInStr+j+2]);

	   Distance=a+b+c;
	   SelectedCluster=0;

	   for(int k=1; k<ClusterCount; k++)		
		{
			a=(DevClusterCore[k*PIXELSIZE]-Image[i*ElementsInStr+j])*(DevClusterCore[k*PIXELSIZE]-Image[i*ElementsInStr+j]);
			b=(DevClusterCore[k*PIXELSIZE+1]-Image[i*ElementsInStr+j+1])*(DevClusterCore[k*PIXELSIZE+1]-Image[i*ElementsInStr+j+1]);
			c=(DevClusterCore[k*PIXELSIZE+2]-Image[i*ElementsInStr+j+2])*(DevClusterCore[k*PIXELSIZE+2]-Image[i*ElementsInStr+j+2]);

			if((a+b+c)<Distance)
			{
		    	Distance=a+b+c;
				SelectedCluster=k;
			}
		}

	   if(Image[i*ElementsInStr+j+3]!=SelectedCluster) (*LastIter)=0;  
		Image[i*ElementsInStr+j+3]=SelectedCluster;


	}
}

__host__ void ClusterDustribution(unsigned char* Image, int ClusterCount, int Width, int Height, int* LastIter)
{
	int ElementsInStr=PIXELSIZE*Width;
	int Distance=0;
	int SelectedCluster=0;
	int a,b,c;  

	*LastIter=1;

	for(int i=0; i<Height; i++)
	{
		for(int j=0; j<ElementsInStr; j+=PIXELSIZE)
		{
			a=(ClusterCore[0]-Image[i*ElementsInStr+j])*(ClusterCore[0]-Image[i*ElementsInStr+j]);
			b=(ClusterCore[1]-Image[i*ElementsInStr+j+1])*(ClusterCore[1]-Image[i*ElementsInStr+j+1]);
			c=(ClusterCore[2]-Image[i*ElementsInStr+j+2])*(ClusterCore[2]-Image[i*ElementsInStr+j+2]);

            Distance=a+b+c;
			SelectedCluster=0;
			for(int k=1; k<ClusterCount; k++)		
			{
				a=(ClusterCore[k*PIXELSIZE]-Image[i*ElementsInStr+j])*(ClusterCore[k*PIXELSIZE]-Image[i*ElementsInStr+j]);
			    b=(ClusterCore[k*PIXELSIZE+1]-Image[i*ElementsInStr+j+1])*(ClusterCore[k*PIXELSIZE+1]-Image[i*ElementsInStr+j+1]);
				c=(ClusterCore[k*PIXELSIZE+2]-Image[i*ElementsInStr+j+2])*(ClusterCore[k*PIXELSIZE+2]-Image[i*ElementsInStr+j+2]);

				if((a+b+c)<Distance)
				{
					Distance=a+b+c;
					SelectedCluster=k;
				}
			}
			if(Image[i*ElementsInStr+j+3]!=SelectedCluster) *LastIter=0;  
			Image[i*ElementsInStr+j+3]=SelectedCluster;
		}	
	}

}

__host__ void ClusterOffset(unsigned char* Image, int ClusterCount, int Width, int Height) 
{
	for(int i=0; i<ClusterCount; i++)
		ClusterPixelCount[i]=0;

	for(int i=0; i<4*ClusterCount; i++)
		ClusterColorSum[i]=0;

	int ElementsInStr=PIXELSIZE*Width;
	int ClusterNum=0;

	for(int i=0; i<Height; i++)
	{
		for(int j=0; j<ElementsInStr; j+=PIXELSIZE)
		{
			ClusterNum=Image[i*ElementsInStr+j+3];
			ClusterPixelCount[ClusterNum]++;
			ClusterColorSum[ClusterNum*PIXELSIZE]+=Image[i*ElementsInStr+j];
			ClusterColorSum[ClusterNum*PIXELSIZE+1]+=Image[i*ElementsInStr+j+1];
			ClusterColorSum[ClusterNum*PIXELSIZE+2]+=Image[i*ElementsInStr+j+2];		    
		}
	}

	for(int i=0; i<ClusterCount; i++)
	{
		ClusterCore[i*PIXELSIZE]=ClusterColorSum[i*PIXELSIZE]/ClusterPixelCount[i];	
		ClusterCore[i*PIXELSIZE+1]=ClusterColorSum[i*PIXELSIZE+1]/ClusterPixelCount[i];	
		ClusterCore[i*PIXELSIZE+2]=ClusterColorSum[i*PIXELSIZE+2]/ClusterPixelCount[i];	
	}
}


int main()
{
   char InPath[256];
   char OutPath[256];

   scanf("%s", InPath);

   FILE* InPut = fopen(InPath, "rb");
    if (InPut == NULL)
    {
        fprintf(stderr, "Cannot open in.data");
        exit(0);
    }

	scanf("%s", OutPath);
	FILE* OutPut = fopen(OutPath, "wb");
    if (OutPut == NULL)
    {
        fprintf(stderr, "Cannot create out.data");
        exit(0);
    }

	int ClusterNumber;     

	scanf("%d", &ClusterNumber);

	int* Xcoords = (int*)malloc(ClusterNumber*sizeof(int)); 
	int* Ycoords = (int*)malloc(ClusterNumber*sizeof(int)); 

	for(int i=0; i<ClusterNumber; i++)
	{
		scanf("%d", &Ycoords[i]);
		scanf("%d", &Xcoords[i]);
	}


    ClusterCore = (double*)malloc(4*ClusterNumber*sizeof(double));
	ClusterPixelCount = (unsigned int*)malloc(ClusterNumber*sizeof(unsigned int));
	ClusterColorSum = (double*)malloc(4*ClusterNumber*sizeof(double));

	int Width;
	int Height;

	fread(&Width, sizeof(int), 1, InPut);
	fread(&Height, sizeof(int), 1, InPut);


	unsigned char* Image = (unsigned char*)malloc(4*Width*Height*sizeof(unsigned char));
	fread(Image, 4*Width*Height*sizeof(unsigned char), 1, InPut);


	unsigned char* Dev_Image;
	cudaMalloc((void**)&Dev_Image, 4*Width*Height*sizeof(unsigned char));
	cudaMemcpy(Dev_Image, Image, 4*Width*Height*sizeof(unsigned char), cudaMemcpyHostToDevice);


	for(int i=0; i<ClusterNumber; i++)
	{
		ClusterCore[i*PIXELSIZE]=Image[4*Width*Xcoords[i]+PIXELSIZE*Ycoords[i]];
		
		ClusterCore[i*PIXELSIZE+1]=Image[4*Width*Xcoords[i]+PIXELSIZE*Ycoords[i]+1];

		ClusterCore[i*PIXELSIZE+2]=Image[4*Width*Xcoords[i]+PIXELSIZE*Ycoords[i]+2];

		ClusterCore[i*PIXELSIZE+3]=0;
	}
	
    int* NotLastIter;     
	int* HostNotLastIter = (int*)malloc(sizeof(int));
	*HostNotLastIter=1;


	cudaMalloc((void**)&NotLastIter, sizeof(int));
	cudaMemcpy(NotLastIter, HostNotLastIter, sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMemcpyToSymbol(DevClusterCore, ClusterCore, 4*ClusterNumber*sizeof(double));

      while(1)
	{
		PixelToCluster<<<128, 512>>>(Dev_Image, ClusterNumber, Width, Height, NotLastIter);
		cudaMemcpy(HostNotLastIter, NotLastIter, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(Image, Dev_Image, 4*Width*Height*sizeof(unsigned char), cudaMemcpyDeviceToHost);

		if((*HostNotLastIter)==1) break;

		ClusterOffset(Image, ClusterNumber, Width, Height);                   
		cudaMemcpyToSymbol(DevClusterCore, ClusterCore, 4*ClusterNumber*sizeof(double));
		*HostNotLastIter=1;
		cudaMemcpy(NotLastIter, HostNotLastIter, sizeof(int), cudaMemcpyHostToDevice);
	}
	

	fwrite(&Width, sizeof(int), 1 ,OutPut);
	fwrite(&Height, sizeof(int), 1, OutPut);
	fwrite(Image, 4*Width*Height*sizeof(unsigned char),1, OutPut);



	cudaFree(NotLastIter);
	cudaFree(Dev_Image);
	free(Image);
	free(ClusterColorSum);
	free(ClusterCore);
	free(ClusterPixelCount);	
    return 0;
}


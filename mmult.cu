#include<stdio.h>
#include<cuda.h>
#include<string>

#define BLOCK_SIZE 16
#define EPSILON 1.0e-15

cudaDeviceProp deviceProp;	

double *m,*v,*result_host,*result_cpu;
double *m_device,*v_device,*result_device;
int     v_length ,mat_row_size , mat_col_size;
int     size = BLOCK_SIZE;

void multiply_cpu() {
	result_cpu = (double*) malloc(mat_row_size * sizeof(double));
	if(result_cpu==NULL) {
		printf("Not enough memory");
		exit(-1);
	}

	int i,j;
	for(i = 0; i < mat_row_size; i++) {
		result_cpu[i]=0;
		for(j = 0; j < mat_col_size; j++)
			result_cpu[i] += m[i*v_length+j] * v[j];
	}
}

double calculate_bandwidth(float & Tsec) {
    float ret = (1.0e-9 * (( size*size + size )/Tsec));
	return ret;
}

void cuda_free_wrapper(double * arr[],int len) {
	for(int i = 0; i < len; i++) cudaFree(arr[i]);
}

/* function to calculate relative error*/
void relative_error(double* device, double* host, int size) {
	double err = 0.0, max_err = 0.0;
	bool flag = false;

	for(int i = 0; i < size; ++i) {

		err = fabs((host[i] - device[i]) )/ max(fabs(host[i]), fabs(device[i]));

		if (err > EPSILON && err != 0.0e+00 ) {       
			max_err = max(max_err, err);
			flag = true;
		}

	}

	if(flag) printf("\n error %e on machine with precision %e\n", max_err, EPSILON);
}

void fill_matrix(double* vec,int size) {
	int ind;
	for(ind = 0; ind < size; ind++) vec[ind] = drand48();
}


__global__ void MatVectMultiplication(double *m, double *v, int mat_row_size, int n, double* result_device) {
	int tidx = blockIdx.x*blockDim.x + threadIdx.x;
	int tidy = blockIdx.y*blockDim.y + threadIdx.y;
	int tindex = tidx + gridDim.x * BLOCK_SIZE * tidy;


	if( tindex < mat_row_size) {
		int t = tindex * n;
		result_device[tindex] = 0.00;
		for(int i = 0; i < n; i++) result_device[tindex] += m[t+i] * v[i];
	}

	__syncthreads();
}



void MatVectMul() {
	int max = BLOCK_SIZE*BLOCK_SIZE;
	int BlocksPerGrid = mat_row_size/max + 1;
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	if(mat_row_size % max == 0) BlocksPerGrid--;
	dim3 dimGrid(1, BlocksPerGrid);

	MatVectMultiplication<<<dimGrid,dimBlock>>>(m_device,v_device,mat_row_size,v_length,result_device);
}


double sim() {
	v_length = mat_col_size = mat_row_size = size;

	float elapsedTime,Tsec;
	cudaEvent_t start,stop;


	m =new double[mat_row_size*mat_col_size];
	v = new double[v_length];
	result_host = new double[mat_row_size];


	if(m == NULL || v == NULL || result_host == NULL) {
		printf("Not enough memory\n");
		exit(-1);
	}

	fill_matrix(m,mat_row_size*mat_col_size);
	fill_matrix(v,v_length);


	cudaEventCreate (&start);
	cudaEventCreate (&stop);

	cudaMalloc( (void**)&m_device, mat_row_size * mat_col_size * sizeof(double));
	cudaMalloc( (void**)&v_device, v_length * sizeof(double));
	cudaMalloc( (void**)&result_device, mat_row_size * sizeof(double));

	cudaMemcpy((void*)m_device, (void*)m, mat_row_size*mat_col_size*sizeof(double) ,cudaMemcpyHostToDevice);
	cudaMemcpy((void*)v_device, (void*)v,v_length*sizeof(double),cudaMemcpyHostToDevice);

	cudaEventRecord (start, 0);

	MatVectMul();

	cudaEventRecord (stop, 0);
	cudaEventSynchronize (stop);
	cudaEventElapsedTime ( &elapsedTime, start, stop);

	Tsec= 1.0e-3*elapsedTime;

	double ret = calculate_bandwidth(Tsec);


	cudaMemcpy((void*)result_host, (void*)result_device, mat_row_size*sizeof(double), cudaMemcpyDeviceToHost);

	multiply_cpu();
	relative_error(result_cpu,result_host,size);
	/*free the memory from GPU */
	double *array[3];
	array[0]=m_device;
	array[1]=v_device;
	array[2]=result_device;
	cuda_free_wrapper(array,3);

	//free host memory----------
	free(m);
	free(v);
	free(result_host);
	free(result_cpu);

	return ret;
}


int main(int argc, char** argv) {
	
	cudaSetDevice(0);

	int startsize, endsize;
	if(argc < 2) startsize = 8;
	else startsize = std::stoi(argv[1]);

	if(argc < 3) endsize = 16384;
	else endsize = std::stoi(argv[2]);

	int device;
	// Current Device
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&deviceProp,device);
	printf("Using device %d: %s \n", device, deviceProp.name);

	printf("Matrix size \t  Bandwith\n");

	for(size = startsize; size <= endsize; size *=2) {
		printf("%d \t \t %f\n", size, sim());
	}

	return 0;
}
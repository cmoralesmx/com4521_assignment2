#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "NBody.h"
#include "NBodyVisualiser.h"

#define USER_NAME "acp15cam"		//replace with your username
#define BUFFER_SIZE 512
#define SOFTENING_2 4.0f
#define DEBUG 0
#define ZERO 1e-6f
#define THREADS_PER_BLOCK 512
#define THREADS_PER_BLOCK2 64
#define THREADS_PER_BLOCK2_2 4096
#define WARP_SIZE 32

struct nbodies{
	float *x, *y, *vx, *vy, *m, *inv_m;
};
struct nbodies h_nbodies, d_nbodies;

void print_help();
void simulate(int iterations);
void step(void);
short operation_mode(const int argc, char *argv[]);
int fileReader(const char *filename);
int readLine(char buffer[], FILE *f);
char* copyString(const char * source);
short prepareData(const char * inputFilename);
char * getFilename(int argc, char *argv[], int secondValidCount, int secondPosition);
void assignDefaultValues(nbody *row);
void assignDefaultValuesSOA(int i);
void generateRandomData();
void displayData();
__global__ void parallelOverBodies(nbodies d_nbodies, float * activityMap, const int numberOfBodies, const float gridLimit, const unsigned short gridDimmension);
short allocateDeviceMemory();
__global__ void updateActivityMap(float * activityMap, const float inverse_numberOfBodies, const unsigned short gridDimmension, const unsigned short n2);
__global__ void body2body(float l_x, float l_y, float * s_x, float * s_y, float * s_vx, float * s_vy);
__global__ void sum_warp_kernel_shfl_down(float *a);
__global__ void parallelBody2Body(nbodies d_nbodies, float * d_activityMap, const int numberOfBodies, const float inv_gridLimit, const unsigned short gridDimm, const dim3 blocksPerGrid, const dim3 threadsPerBlock);
MODE mode;
int numberOfBodies;
float inverse_numberOfBodies;
short gridDimmension;
float * activityMap, gridLimit, *d_activityMap;
time_t t;


int main(int argc, char *argv[]) {
	int iterations;
	float seconds = -1.0f;
	
	srand((unsigned)time(NULL));
	clock_t begin = clock(), end = clock();

	// Check the received parameters follow the stablished format
	// and find if this is a simulation or visualisation
	short opMode = operation_mode(argc - 1, argv + 1);

	if ( opMode == -1 ){
		printf("Wrong parameters provided\n");
		print_help();
		return 0;
	}
	else {
		numberOfBodies = atoi(argv[1]); // N
		inverse_numberOfBodies = 1.0f / numberOfBodies;
		gridDimmension = atoi(argv[2]); // D
		// calculate the ranges for the "bins" of the grid
		gridLimit = 1.0f / (gridDimmension - 1);

		if (DEBUG){
			printf("Number of bodies: %d\n", numberOfBodies);
			printf("Activity grid dimmension: %d\n", gridDimmension);
		}

		// Allocate heap memory
		//data = (nbody*)malloc(sizeof(nbody) * numberOfBodies);
		size_t size = sizeof(float) * numberOfBodies;
		h_nbodies.x = (float*)malloc(size);
		h_nbodies.y = (float*)malloc(size);
		h_nbodies.vx = (float*)malloc(size);
		h_nbodies.vy = (float*)malloc(size);
		h_nbodies.m = (float*)malloc(size);
		h_nbodies.inv_m = (float*)malloc(size);
		activityMap = (float*)malloc(sizeof(float) * gridDimmension * gridDimmension);
		for (int v = 0; v < gridDimmension * gridDimmension; v++)
			activityMap[v] = 0;

		if (stricmp(argv[3], "CPU") == 0)
			mode = CPU;
		else if
			(stricmp(argv[3], "OPENMP") == 0)
				mode = OPENMP;
		else
		{
			mode = CUDA;
			allocateDeviceMemory();
		}

		if (DEBUG)
			printf("%s, %s mode\n", opMode == 0 ? "Simulation" : "Visualization", argv[3]);
		if (opMode == 0) {
			// simulation
			iterations = atoi(argv[5]);
			if (DEBUG)
				printf("Iterations: %d\n", iterations);
			// prepare simulation data
			// input parameters for simulation allow the filename to be specified in possition 7 for 8 params
			if (prepareData(getFilename(argc, argv, 8, 7)) == 0){
				if (DEBUG)
					printf("Simulating... ");
				
				// perform a fixed number of simulation steps (then output the timing results)
				begin = clock();
				simulate(iterations);
				end = clock();
				seconds = (end - begin) / (float)CLOCKS_PER_SEC;
				
				if (DEBUG)
					printf("Done!\n");
			}
		}
		else {
			// prepare visualisation data
			// input parameters for visualisation allow the filename to be specified in possition 5 for 6 params
			if (prepareData(getFilename(argc, argv, 6, 5)) == 0) {
				// configure and start the visualiser (then output the timing results).
				initViewer(numberOfBodies, gridDimmension, mode, &step);
				if (mode == CUDA){
					setNBodyPositions2f(d_nbodies.x, d_nbodies.y);
					setHistogramData(d_activityMap);
				}
				else{
					setNBodyPositions2f(h_nbodies.x, h_nbodies.y);
					setHistogramData(activityMap);
				}
				begin = clock();
				startVisualisationLoop();
				end = clock();
				seconds = (end - begin) / (float)CLOCKS_PER_SEC;
			}
		}
		if (seconds > -1.0f)
			printf("Execution time %.0f seconds %03.0f milliseconds\n", seconds, (seconds - (int)seconds) * 1000);
		else
			printf("No computation performed?");
		//getchar();*/
	}
	// release heap memory
	free(activityMap);
	free(h_nbodies.x);
	free(h_nbodies.y);
	free(h_nbodies.vx);
	free(h_nbodies.vy);
	free(h_nbodies.m);
	free(h_nbodies.inv_m);
	if (mode == CUDA){
		cudaDeviceReset();
		cudaFree(&d_nbodies.x);
		cudaFree(&d_nbodies.y);
		cudaFree(&d_nbodies.vx);
		cudaFree(&d_nbodies.vy);
		cudaFree(&d_nbodies.m);
		cudaFree(&d_nbodies.inv_m);
		cudaFree(&d_activityMap);
	}
	
	return 0;
}

short allocateDeviceMemory(){
	// allocate device dynamic global memory
	cudaError_t cudaStatus1 = cudaMalloc((void **)&d_nbodies.x, sizeof(d_nbodies.x) * numberOfBodies);
	cudaError_t cudaStatus2 = cudaMalloc((void **)&d_nbodies.y, sizeof(d_nbodies.y) * numberOfBodies);
	cudaError_t cudaStatus3 = cudaMalloc((void **)&d_nbodies.vx, sizeof(d_nbodies.vx) * numberOfBodies);
	cudaError_t cudaStatus4 = cudaMalloc((void **)&d_nbodies.vy, sizeof(d_nbodies.vy) * numberOfBodies);
	cudaError_t cudaStatus5 = cudaMalloc((void **)&d_nbodies.m, sizeof(d_nbodies.m) * numberOfBodies);
	cudaError_t cudaStatus6 = cudaMalloc((void **)&d_nbodies.inv_m, sizeof(d_nbodies.inv_m) * numberOfBodies);
	if (cudaStatus1 == cudaSuccess && cudaStatus2 == cudaSuccess && cudaStatus3 == cudaSuccess
		&& cudaStatus4 == cudaSuccess && cudaStatus5 == cudaSuccess && cudaStatus6 == cudaSuccess)
	{
		cudaError_t cudaStatus = cudaMalloc((void**)&d_activityMap, sizeof(float) * gridDimmension * gridDimmension);
		if (cudaStatus == cudaSuccess)
		{
			cudaStatus = cudaMemcpy(d_activityMap, activityMap, sizeof(float) * gridDimmension * gridDimmension, cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess){
				printf("Error copying the activityMap data to device memory");
				return 1;
			}
		}
		else{
			printf("Error allocating CUDA device memory for the activityMap");
			return 1;
		}
	}
	else{
		printf("Error allocating CUDA device memory for the data");
		return 1;
	}
	return 0;
}
char * getFilename(int argc, char *argv[], int secondValidCount, int secondPosition){
	if (argc == 7){
		return copyString(argv[6]);
	}
	else if (argc == secondValidCount){
		return copyString(argv[secondPosition]);
	}
	return NULL;
}

/*
input
const char * inputFilename - full path to the input file
output
-1	a line read does not match the required format
1	the number of records and number of bodies does not match
0	process completed successfully
*/
short loadOrGenerateData(const char * inputFilename){
	if (inputFilename != NULL){
		// read data from file
		return fileReader(inputFilename);
	}
	else {
		// generate random data.
		generateRandomData();
		return 0;
	}
}

/*
input 
const char * inputFilename - full path to the input file
output
-1	a line read does not match the required format
1	the number of records and number of bodies does not match
0	process completed successfully
-2	CUDA error copying data to device
*/
short prepareData(const char * inputFilename){
	short loadDataErrors = loadOrGenerateData(inputFilename);
	if (loadDataErrors != 0){
		if (DEBUG)
			printf("ERROR loading or generating data!\n");
	}
	if (mode == CUDA){
		// copy host data to device
		cudaError_t cudaStatus1, cudaStatus2, cudaStatus3, cudaStatus4, cudaStatus5, cudaStatus6;
		size_t size = sizeof(float) * numberOfBodies;
		cudaStatus1 = cudaMemcpy(d_nbodies.x, h_nbodies.x, size, cudaMemcpyHostToDevice);
		cudaStatus2 = cudaMemcpy(d_nbodies.y, h_nbodies.y, size, cudaMemcpyHostToDevice);
		cudaStatus3 = cudaMemcpy(d_nbodies.vx, h_nbodies.vx, size, cudaMemcpyHostToDevice);
		cudaStatus4 = cudaMemcpy(d_nbodies.vy, h_nbodies.vy, size, cudaMemcpyHostToDevice);
		cudaStatus5 = cudaMemcpy(d_nbodies.m, h_nbodies.m, size, cudaMemcpyHostToDevice);
		cudaStatus6 = cudaMemcpy(d_nbodies.inv_m, h_nbodies.inv_m, size, cudaMemcpyHostToDevice);
		if (cudaStatus1 != cudaSuccess || cudaStatus2 != cudaSuccess || cudaStatus3 != cudaSuccess
			|| cudaStatus4 != cudaSuccess || cudaStatus5 != cudaSuccess || cudaStatus6 != cudaSuccess){
			if (DEBUG)
				printf("ERROR copying host data to device\n");
			return -2;
		}
	}
	return loadDataErrors;
}

void displayData(){
	for (int i = 0; i < numberOfBodies; i++){
		if (numberOfBodies < 10 || i < 5 || i > numberOfBodies - 5)
			printf("[%d] x=%f, y=%f, vx=%f, vy=%f, mass=%f\n", i, h_nbodies.x[i],
				h_nbodies.y[i], h_nbodies.vx[i], h_nbodies.vy[i], h_nbodies.m[i]);
	}
}
/* function to simplify memory allocation and content copy
	taken from C p.88
	Tried to implement the pointer version of strcopy from the book but does
	not work so I had to resort to the subscript version
*/
char* copyString(const char * source){
	int i = 0;
	char * dest = (char *)malloc(strlen(source) * sizeof(char));
	
	//while (*dest++ = *source++); // this method does not work, why?
	while (dest[i] = source[i])
		i++;
	return dest;
}

void simulate(int iterations)
{
	for (int iteration = 0; iteration < iterations; iteration++){
		step();
	}
}

/*
Perform a simulation step of the system
*/
void step(void)
{
	int i, j;

	switch (mode)
	{
	case CPU:
		for (i = 0; i < numberOfBodies; i++)
		{
			float euclidean_x, euclidean_y, soft_norm, force_x, force_y;
			float sum_x = 0, sum_y = 0;

			for (j = 0; j < numberOfBodies; j++){
				// m_j (x_j - x_i) / (|| x_j - x_i ||^2 + softening^2 )^(3/2)
				euclidean_x = h_nbodies.x[j] - h_nbodies.x[i];
				euclidean_y = h_nbodies.y[j] - h_nbodies.y[i];
				soft_norm = (float)pow(euclidean_x * euclidean_x + euclidean_y * euclidean_y + SOFTENING_2, 1.5f) + ZERO;
				// this simation is independent for x or y
				sum_x += (h_nbodies.m[j] * euclidean_x) / soft_norm;
				sum_y += (h_nbodies.m[j] * euclidean_y) / soft_norm;
			}
			// Calculate the force
			// F_i = G * m_i * sum
			force_x = G * h_nbodies.m[i] * sum_x;
			force_y = G * h_nbodies.m[i] * sum_y;

			// simulate the movement

			// calculate the position
			// WE DO THIS FIRST due to its dependance on current velocity
			// x_t+1 = x_t + dt * v_t
			h_nbodies.x[i] += dt * h_nbodies.vx[i];
			h_nbodies.y[i] += dt * h_nbodies.vy[i];

			// update the velocity value 
			// acceleration is also computed here, no need for independent computation
			// v_t+1 = v_t + dt * a  // acceleration a_i = F_i / m_i
			h_nbodies.vx[i] += dt * (force_x / (h_nbodies.m[i] + ZERO));
			h_nbodies.vy[i] += dt * (force_y / (h_nbodies.m[i] + ZERO));

			/*
			compute the position for a body in the activityMap and increase the
			corresponding body count
			index computed according to "The C programming guide" 2nd ed pp.113
			*/
			int col = (int)(h_nbodies.x[i] / (gridLimit + ZERO));
			int row = (int)(h_nbodies.y[i] / (gridLimit + ZERO));
			int cell = (int)(gridDimmension * row + col);
			activityMap[cell] += 1.0f;
		}
		// Now traverse the activityMap to normalise the counts
		// to achieve the intended visualization
		for (i = 0; i < gridDimmension * gridDimmension; i++)
		{
			activityMap[i] /= (float)numberOfBodies;
			activityMap[i] *= gridDimmension;
		}
		break;
	case OPENMP:
//#pragma omp parallel num_threads(4)
	{
		//#pragma omp parallel for default(none) shared(data, activityMap, numberOfBodies, gridLimit, gridDimmension)
		for (i = 0; i < numberOfBodies; i++)
		{
			float euclidean_x, euclidean_y, soft_norm, force_x, force_y;
			float sum_x = 0, sum_y = 0;

#pragma omp parallel for reduction(+: sum_x, sum_y) default(none) shared(h_nbodies, activityMap, numberOfBodies, i, gridLimit, gridDimmension) private (euclidean_x, euclidean_y, soft_norm)
			for (j = 0; j < numberOfBodies; j++)
			{
				// m_j (x_j - x_i) / (|| x_j - x_i ||^2 + softening^2 )^(3/2)
				euclidean_x = h_nbodies.x[j] - h_nbodies.x[i];
				euclidean_y = h_nbodies.y[j] - h_nbodies.y[i];
				soft_norm = (float)pow(euclidean_x * euclidean_x + euclidean_y * euclidean_y + SOFTENING_2, 1.5f) + ZERO;
				// this simation is independent for x or y
				sum_x += (h_nbodies.m[j] * euclidean_x) / soft_norm;
				sum_y += (h_nbodies.m[j] * euclidean_y) / soft_norm;
			}
			// Calculate the force
			// F_i = G * m_i * sum
			force_x = G * h_nbodies.m[i] * sum_x;
			force_y = G * h_nbodies.m[i] * sum_y;

			// simulate the movement

			// calculate the position
			// WE DO THIS FIRST due to its dependance on current velocity
			// x_t+1 = x_t + dt * v_t
			h_nbodies.x[i] += dt * h_nbodies.vx[i];
			h_nbodies.y[i] += dt * h_nbodies.vy[i];

			// update the velocity value 
			// acceleration is also computed here, no need for independent computation
			// v_t+1 = v_t + dt * a  // acceleration a_i = F_i / m_i
			h_nbodies.vx[i] += dt * (force_x / (h_nbodies.m[i] + ZERO));
			h_nbodies.vy[i] += dt * (force_y / (h_nbodies.m[i] + ZERO));

			/*
			compute the position for a body in the activityMap and increase the
			corresponding body count
			index computed according to "The C programming guide" 2nd ed pp.113
			*/
			int col = (int)(h_nbodies.x[i] / (gridLimit + ZERO));
			int row = (int)(h_nbodies.y[i] / (gridLimit + ZERO));
			int cell = (int)(gridDimmension * row + col);
#pragma omp atomic
			activityMap[cell] += 1.0f;
		}
#pragma omp parallel for schedule(dynamic) default(none) shared(gridDimmension, activityMap, numberOfBodies)
		for (i = 0; i < gridDimmension * gridDimmension; i++){
			activityMap[i] /= (float)numberOfBodies;
			activityMap[i] *= gridDimmension;
		}
	}
		break;
	case CUDA:
		// launch the bodies kernel
		dim3 blocksPerGrid((numberOfBodies + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
		dim3 threadsPerBlock(THREADS_PER_BLOCK, 1);
		
		// First CUDA option
		//parallelOverBodies << < blocksPerGrid, threadsPerBlock >> >(d_nbodies, d_activityMap, numberOfBodies, 1.0f/gridLimit, gridDimmension);
		// second CUDA option
		parallelBody2Body << <blocksPerGrid, threadsPerBlock >> >(d_nbodies, activityMap, numberOfBodies, 1.0f / gridLimit, gridDimmension, blocksPerGrid, threadsPerBlock);
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			printf("CUDA error in bodies kernel\n");
		
		// sumation over shared
		// pre-update activity matrix???

		// launch the activity map updater kernel
		updateActivityMap << < blocksPerGrid, threadsPerBlock >> >(d_activityMap, inverse_numberOfBodies, gridDimmension, gridDimmension * gridDimmension);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			printf("CUDA error in activity map kernel\n");

		break;
	}
}
__global__ void updateActivityMap(float * d_activityMap, const float inverse_numberOfBodies, const unsigned short gridDimmension, const unsigned short n2){
	unsigned short idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n2){
		d_activityMap[idx] *= inverse_numberOfBodies;
		d_activityMap[idx] *= gridDimmension;
	}
}
/*
parallelOverBodies - The kernel computes the affecting forces per body.
*/
__global__ void parallelOverBodies(nbodies d_nbodies, float * d_activityMap, const int numberOfBodies, const float inv_gridLimit, const unsigned short gridDimm){
	unsigned short idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numberOfBodies){
		float4 force = { 0.0f, 0.0f, 0.0f, 0.0f };
		nbody body = { d_nbodies.x[idx], d_nbodies.y[idx], d_nbodies.vx[idx], d_nbodies.vy[idx], d_nbodies.m[idx] };
		for (short block = 0; block < gridDim.x; block++){
			__shared__ float s_x[THREADS_PER_BLOCK];
			__shared__ float s_y[THREADS_PER_BLOCK];
			__shared__ float s_m[THREADS_PER_BLOCK];
			unsigned short tid = block * blockDim.x + threadIdx.x;
			s_x[threadIdx.x] = d_nbodies.x[tid];
			s_y[threadIdx.x] = d_nbodies.y[tid];
			s_m[threadIdx.x] = d_nbodies.m[tid];
			__syncthreads();

			for (int j = 0; j < THREADS_PER_BLOCK; j++){
				// m_j (x_j - x_i) / (|| x_j - x_i ||^2 + softening^2 )^(3/2)
				float distance_x = s_x[j] - body.x;
				float distance_y = s_y[j] - body.y;
				
				float dx_2 = distance_x * distance_x;
				float dy_2 = distance_y * distance_y;
				float subt = dx_2 + dy_2;
				// CUDA reciprocal squared root. faster than 1/sqrt(x)
				float inv_sqrt = rsqrtf(subt + SOFTENING_2);
				float inv_sqrt_2 = inv_sqrt * inv_sqrt;
				float inv_sqrt_3 =  inv_sqrt_2 * inv_sqrt;
				// this sumation is independent for x or y
				float mass_inv_sqrt_3 = s_m[j] * inv_sqrt_3;
				force.z += mass_inv_sqrt_3 * distance_x;
				force.w += mass_inv_sqrt_3 * distance_y;
			}
			__syncthreads();
		}
		// Calculate the force
		// F_i = G * m_i * sum
		force.x = G * body.m * force.z;
		force.y = G * body.m * force.w;

		// simulate the movement

		// calculate the position
		// WE DO THIS FIRST due to its dependance on current velocity
		// x_t+1 = x_t + dt * v_t
		d_nbodies.x[idx] += dt * body.vx;
		d_nbodies.y[idx] += dt * body.vy;

		// update the velocity value 
		// acceleration is also computed here, no need for independent computation
		// v_t+1 = v_t + dt * a  // acceleration a_i = F_i / m_i
		d_nbodies.vx[idx] += dt * force.x * d_nbodies.inv_m[idx];
		d_nbodies.vy[idx] += dt * force.y * d_nbodies.inv_m[idx];

		/*
		compute the position for a body in the activityMap and increase the
		corresponding body count
		index computed according to "The C programming guide" 2nd ed pp.113
		*/
		unsigned short col = d_nbodies.x[idx] * inv_gridLimit;
		unsigned short row = d_nbodies.y[idx] * inv_gridLimit;
		unsigned short cell = gridDimm * row + col;
		
		atomicAdd(&d_activityMap[cell], 1.0f);
	}
}

/*
this kernel should compute, the amount of effect each body in matrix B has over the bodies in matrix A
s_accum_vx[padding + threadIdx.x]
s_accum_vy[padding + threadIdx.x]
*/
__global__ void body2body(float l_x, float l_y,
	float * s_x, float * s_y, 
	float * s_vx, float * s_vy){

//	unsigned short itid = blockIdx.x * blockDim.x + threadIdx.x;

	float distance_x = s_x[threadIdx.x] - l_x;
	float distance_y = s_y[threadIdx.x] - l_y;

	float dx_2 = distance_x * distance_x;
	float dy_2 = distance_y * distance_y;
	float subt = dx_2 + dy_2;
	// CUDA reciprocal squared root. faster than 1/sqrt(x)
	float inv_sqrt = rsqrtf(subt + SOFTENING_2);
	float inv_sqrt_3 = inv_sqrt * inv_sqrt * inv_sqrt;
	// this sumation is independent for x or y
	//float mass_inv_sqrt_3 = l_m * inv_sqrt_3;
	s_vx[threadIdx.x] = inv_sqrt_3 * distance_x;
	s_vy[threadIdx.x] = inv_sqrt_3 * distance_y;
}

__global__ void sum_warp_kernel_shfl_down(float *a)
{
	float local_sum = a[threadIdx.x + blockIdx.x * blockDim.x];
	for (int offset = WARP_SIZE / 2; offset>0; offset /= 2)
		local_sum += __shfl_down(local_sum, offset);
	if (threadIdx.x % WARP_SIZE == 0){
		//printf("Warp max is %d", local_sum);
		a[0] = local_sum;
	}
}

/*
parallelBody2Body - This kernel computes the body-to-body interactions.

*/
__global__ void parallelBody2Body(nbodies d_nbodies, float * d_activityMap, const int numberOfBodies, 
	const float inv_gridLimit, const unsigned short gridDimm, const dim3 blocksPerGrid, const dim3 threadsPerBlock){
	unsigned short tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < numberOfBodies){
		// shared memory containers to use as the base for computation
		__shared__ float s_x[THREADS_PER_BLOCK];
		__shared__ float s_y[THREADS_PER_BLOCK];
		__shared__ float s_vx[THREADS_PER_BLOCK];
		__shared__ float s_vy[THREADS_PER_BLOCK];
		__shared__ float s_m[THREADS_PER_BLOCK];
		__shared__ float s_inv_m[THREADS_PER_BLOCK];
		// l_* myPosition
		//float l_x = d_nbodies.x[tid], l_y = d_nbodies.y[tid], l_m = d_nbodies.m[tid];
		// temp accumulator for the velocity
		float accum_vx = 0.0f;
		float accum_vy = 0.0f;

		// initiate te values
		s_x[threadIdx.x] = d_nbodies.x[tid];
		s_y[threadIdx.x] = d_nbodies.y[tid];
		s_m[threadIdx.x] = d_nbodies.m[tid];
		s_inv_m[threadIdx.x] = d_nbodies.inv_m[tid];
		for (int i = 0, tile = 0; i < numberOfBodies; i += THREADS_PER_BLOCK, tile++){
			//unsigned short idx = tile * blockDim.x + threadIdx.x;
			for (int z = 0; z < THREADS_PER_BLOCK; z++){
				s_vx[z] = 0.0f;
				s_vy[z] = 0.0f;
			}
			__syncthreads();
			// launch 1 block iteratively to compute the body 2 body interactions
			// should pass single values: s_x[tile], s_y[tile]
			// pointers to address to begin reading the array at: d_nbodies.x[tile * blockDim.x], d_nbodies.y[tile * blockDim.x]
			// pointers to array: s_vx, s_vy
			body2body << <1, threadsPerBlock >> >(s_x[threadIdx.x], s_y[threadIdx.x],
				&d_nbodies.x[tile * blockDim.x], &d_nbodies.x[tile * blockDim.x], s_vx, s_vy);
			__syncthreads();
			// shuffle warp sum the computed values for this block
			sum_warp_kernel_shfl_down << <blocksPerGrid, threadsPerBlock, 1 >> >(s_vx);
			sum_warp_kernel_shfl_down << <blocksPerGrid, threadsPerBlock, 2 >> >(s_vy);
			// accumulate the velocity values
			accum_vx += s_vx[0];
			accum_vy += s_vy[0];
		}
		// update the rest of the values
		// Calculate the force
		// F_i = G * m_i * sum
		float force_x = G * s_m[threadIdx.x] * accum_vx;
		float force_y = G * s_m[threadIdx.x] * accum_vy;

		// simulate the movement
		// these values should be output to the accumulator array for further reduction

		// calculate the new position for this body
		// x_t+1 = x_t + dt * v_t
		s_x[threadIdx.x] += dt * accum_vx;
		s_y[threadIdx.x] += dt * accum_vy;

		// update the velocity value 
		// acceleration is also computed here, no need for independent computation
		// v_t+1 = v_t + dt * a  // acceleration a_i = F_i / m_i
		s_vx[threadIdx.x] += dt * force_x * s_inv_m[tid]; // these seem to need sumation at the end
		s_vy[threadIdx.x] += dt * force_y * s_inv_m[tid]; // needs to be fixed?

		// save the values to global memory
		d_nbodies.x[tid] = s_x[threadIdx.x];
		d_nbodies.y[tid] = s_y[threadIdx.x];
		d_nbodies.vx[tid] = s_vx[threadIdx.x];
		d_nbodies.vy[tid] = s_vy[threadIdx.x];
		///*
		//compute the position for a body in the activityMap and increase the
		//corresponding body count
		//index computed according to "The C programming guide" 2nd ed pp.113
		//*/
		unsigned short col = s_x[threadIdx.x] * inv_gridLimit;
		unsigned short row = s_y[threadIdx.x] * inv_gridLimit;
		unsigned short cell = gridDimm * row + col;

		atomicAdd(&d_activityMap[cell], 1.0f);

	}
}
void print_help(){
	printf("nbody_%s N D M [-i I] [-i input_file]\n", USER_NAME);

	printf("where:\n");
	printf("\tN                Is the number of bodies to simulate.\n");
	printf("\tD                Is the integer dimension of the activity grid. The Grid has D*D locations.\n");
	printf("\tM                Is the operation mode, either  'CPU', 'OPENMP' or 'CUDA'\n");
	printf("\t[-i I]           Optionally specifies the number of simulation iterations 'I' to perform. \n\t\t\t\tSpecifying no value will use visualisation mode. \n");
	printf("\t[-f input_file]  Optionally specifies an input file with an initial N bodies of data. \n\t\t\t\tIf not specified random data will be created.\n");
}

/**
Validate the inputs provided in comliance with print_help function
input:
int argc
char *argv[]
returns:
-1 wrong parameters provided
1 parameters specify visualisation run
0 parameters specify simulation run
*/
short operation_mode(const int argc, char **argv){
	if (argc < 3 || argc > 7){
		return -1;
	}
	if (argc < 5){
		// This number of parameters can only match a visualisation run

		// check for integer values in first parameters
		if (atoi(argv[0]) <= 0 || atoi(argv[1]) <= 0)
			return -1;
		// parameter 3 only CPU or OPENMP valid
		if (stricmp(argv[2], "CPU") != 0 && stricmp(argv[2], "OPENMP") != 0 && stricmp(argv[2], "CUDA") != 0)
			return -1;

		// 5th parameter must be '-i', if present
		if (argc == 5 && stricmp(argv[4], "-i") != 0)
			return -1;

		// parameters seem to comply
		return 1;
	}
	else {
		// check for integer values in first parameters
		if (atoi(argv[0]) == 0 || atoi(argv[1]) == 0)
			return -1;
		// parameter 3 only CPU or OPENMP valid
		if (stricmp(argv[2], "CPU") != 0 && stricmp(argv[2], "OPENMP") != 0 && stricmp(argv[2], "CUDA") != 0)
			return -1;

		// for 6 params, there must be a numeric value in param 5
		// params -i -f can not be specified both w.o. parameters at the same time
		if (argc == 5)
			if (stricmp(argv[3], "-f") == 0)
				// iterations were skipped but the file was provided
				// this is an unlimited visualisation
				return 1;
			else
				if (atoi(argv[4]) == 0)
					return -1;
				else
					// this is a simulation with N iteration specified
					return 0;

		// for 7 params, the last one can only be the path to the input file
		if (argc == 6)
			if (stricmp(argv[5], "-f") == 0)
				return -1;
			else
				// this is a visualistation with an input file specified
				return 1;

		// Number of simulation cannot be 0
		if (argc == 7 && atoi(argv[4]) == 0)
			return -1;

		// this is a simulation with full parameters specified. correctly?
		return 0;
	}
}
/*
Counts the number of commas in the buffer
input:
const char * buffer - the buffer to use as source
output:
int the number of commas
*/
short countCommas(const char * buffer){
	unsigned int i;
	unsigned short commas = 0;
	// Check that the line contains 4 commas
	for (i = 0; i < strlen(buffer); i++){
		if (buffer[i] == ',')
			++commas;
	}
	return commas;
}
/*
Executes a loop to fill the data structure with default parameters
*/
void generateRandomData(){
	if (DEBUG)
		printf("Generating random data for %d bodies. ", numberOfBodies);
	for (int i = 0; i < numberOfBodies; i++)
		assignDefaultValuesSOA(i);
	if (DEBUG)
		printf("Done.\n");
}
/*
Assigns default values in accordance to specifications
x,y = random [0,1]
vx,vy = 0
mass = 1/N
input:
float* row - pointer to the array to fill
const int N - number of expected bodies
return:
void
*/
void assignDefaultValues(nbody *row){
	//(double)rand() / (double)((unsigned)RAND_MAX + 1)
	row->x = (float)((double)rand() / (double)((unsigned)RAND_MAX + 1));
	row->y = (float)((double)rand() / (double)((unsigned)RAND_MAX + 1));
	if (row->x < 0.000001)
		row->x = 0;
	if (row->y < 0.000001) 
		row->y = 0;
	row->vx = 0;
	row->vy = 0;
	row->m = 1.0f / (float)numberOfBodies;
}
void assignDefaultValuesSOA(int i){
	//(double)rand() / (double)((unsigned)RAND_MAX + 1)
	h_nbodies.x[i] = (float)((double)rand() / (double)((unsigned)RAND_MAX + 1));
	h_nbodies.y[i] = (float)((double)rand() / (double)((unsigned)RAND_MAX + 1));
	if (h_nbodies.x[i] < 0.000001)
		h_nbodies.x[i] = 0;
	if (h_nbodies.y[i] < 0.000001)
		h_nbodies.y[i] = 0;
	h_nbodies.vx[i] = 0;
	h_nbodies.vy[i] = 0;
	h_nbodies.m[i] = 1.0f / (float)numberOfBodies;
	h_nbodies.inv_m[i] = 1.0f / h_nbodies.m[i];
}
/*
Loads the data from specified input file.
Fills any value not provided with default values in accordance to specifications
input:
const char * filename - pointer to the file to read
const int N - number of bodies expected in the file
float ** data - pre allocated 2d container array for the data to loadS
output:
-1	a line read does not match the required format (4 commas)
1	the number of records in and number of bodies does not match
0	process completed successfully
*/
int fileReader(const char *filename){
	char buffer[BUFFER_SIZE];
	int body_count = 0;

	if (DEBUG)
		printf("Input file: %s\n", filename);

	FILE *f = fopen(filename, "r");

	while (readLine(buffer, f)){
		
		if (buffer[0] == '#') // comment lines are ignored
			continue;

		if (countCommas(buffer) != 4){
			// the line read does not follow the required format
			fclose(f);
			return -1;
		}

		assignDefaultValuesSOA(body_count);

		if (body_count < numberOfBodies){
			// valid format: 0.5f, 0.5f, 0.0f, 0.0f, 0.1f
			sscanf(buffer, "%ff, %ff, %ff, %ff, %ff", &h_nbodies.x[body_count],
				&h_nbodies.y[body_count], &h_nbodies.vx[body_count],
				&h_nbodies.vy[body_count], &h_nbodies.m[body_count]);
			h_nbodies.inv_m[body_count] = 1.0f / h_nbodies.m[body_count];
			++body_count;
		}
	}
	fclose(f);
	if (body_count != numberOfBodies){
		printf("Number of bodies in input file does not match the parameter specified\n");
		return 1;
	}
	return 0;
}
/* Reads a single line from the specified file
input:
char buffer[] - the buffer to store the read characters
const FILE *f - a pointer to the file to read
output:
0 upon reaching the EOF indicator
1 for a line successfully read
*/
int readLine(char buffer[], FILE *f){
	unsigned short i = 0;
	char c = 0;

	while ((c = getc(f)) != '\n'){
		if (c == EOF)
			return 0;
		// Add character to buffer
		buffer[i++] = c;
		// Check index for overflow
		if (i == BUFFER_SIZE){
			fprintf(stderr, "buffer overflow");
			exit(0);
		}
	}
	// Ensure the buffer is correctly terminated
	buffer[i] = '\0';
	
	return 1;
}
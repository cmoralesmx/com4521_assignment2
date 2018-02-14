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

struct nbodies{
	float *x, *y, *vx, *vy, *m;
};
struct nbodies d_nbodies, h_nbodies;

void print_help();
void simulate(int iterations);
void step(void);
int operation_mode(const int argc, char *argv[]);
int fileReader(const char *filename);
int readLine(char buffer[], FILE *f);
char* copyString(const char * source);
int prepareData(const char * inputFilename);
char * getFilename(int argc, char *argv[], int secondValidCount, int secondPosition);
void assignDefaultValues(nbody *row);
void assignDefaultValuesSOA(int i);
void generateRandomData();
void displayData();
__global__ void parallelOverBodies(nbodies d_nbodies, float * activityMap, int numberOfBodies, float gridLimit, int gridDimmension);
int allocateDeviceMemory();
__global__ void updateActivityMap(float * activityMap, int numberOfBodies, int gridDimmension);

MODE mode;
//nbody * data, * d_data;
int numberOfBodies, gridDimmension;
float * activityMap, gridLimit, * d_activityMap;
time_t t;


int main(int argc, char *argv[]) {
	int iterations;
	float seconds = -1.0f;

	srand((unsigned)time(NULL));
	clock_t begin = clock(), end = clock();

	// Check the received parameters follow the stablished format
	// and find if this is a simulation or visualisation
	int opMode = operation_mode(argc - 1, argv + 1);

	if ( opMode == -1 ){
		printf("Wrong parameters provided\n");
		print_help();
		return 0;
	}
	else {
		numberOfBodies = atoi(argv[1]); // N
		gridDimmension = atoi(argv[2]); // D
		// calculate the ranges for the "bins" of the grid
		gridLimit = 1.0f / gridDimmension; // (gridDimmension - 1);

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
			printf("Execution time %.0f seconds %.0f milliseconds\n", seconds, (seconds - (int)seconds) * 1000);
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
	if (mode == CUDA){
		cudaFree(&d_nbodies.x);
		cudaFree(&d_nbodies.y);
		cudaFree(&d_nbodies.vx);
		cudaFree(&d_nbodies.vy);
		cudaFree(&d_nbodies.m);
		cudaFree(d_activityMap);
	}
	
	return 0;
}

int allocateDeviceMemory(){
	// allocate device dynamic global memory
	cudaError_t cudaStatus1 = cudaMalloc((void **)&d_nbodies.x, sizeof(d_nbodies.x) * numberOfBodies);
	cudaError_t cudaStatus2 = cudaMalloc((void **)&d_nbodies.y, sizeof(d_nbodies.y) * numberOfBodies);
	cudaError_t cudaStatus3 = cudaMalloc((void **)&d_nbodies.vx, sizeof(d_nbodies.vx) * numberOfBodies);
	cudaError_t cudaStatus4 = cudaMalloc((void **)&d_nbodies.vy, sizeof(d_nbodies.vy) * numberOfBodies);
	cudaError_t cudaStatus5 = cudaMalloc((void **)&d_nbodies.m, sizeof(d_nbodies.m) * numberOfBodies);
	if (cudaStatus1 == cudaSuccess && cudaStatus2 == cudaSuccess && cudaStatus3 == cudaSuccess
		&& cudaStatus4 == cudaSuccess && cudaStatus5 == cudaSuccess)
	{
		cudaError_t cudaStatus = cudaMalloc((void **)&d_activityMap, sizeof(float) * gridDimmension * gridDimmension);
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
int loadOrGenerateData(const char * inputFilename){
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
int prepareData(const char * inputFilename){
	int loadDataErrors = loadOrGenerateData(inputFilename);
	if (loadDataErrors != 0){
		if (DEBUG)
			printf("ERROR loading or generating data!\n");
	}
	if (mode == CUDA){
		// copy host data to device
		cudaError_t cudaStatus1, cudaStatus2, cudaStatus3, cudaStatus4, cudaStatus5;
		size_t size = sizeof(float) * numberOfBodies;
		cudaStatus1 = cudaMemcpy(d_nbodies.x, h_nbodies.x, size, cudaMemcpyHostToDevice);
		cudaStatus2 = cudaMemcpy(d_nbodies.y, h_nbodies.y, size, cudaMemcpyHostToDevice);
		cudaStatus3 = cudaMemcpy(d_nbodies.vx, h_nbodies.vx, size, cudaMemcpyHostToDevice);
		cudaStatus4 = cudaMemcpy(d_nbodies.vy, h_nbodies.vy, size, cudaMemcpyHostToDevice);
		cudaStatus5 = cudaMemcpy(d_nbodies.m, h_nbodies.m, size, cudaMemcpyHostToDevice);
		if (cudaStatus1 != cudaSuccess || cudaStatus2 != cudaSuccess || cudaStatus3 != cudaSuccess
			|| cudaStatus4 != cudaSuccess || cudaStatus5 != cudaSuccess){
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
		//dim3 blocksPerGrid(numberOfBodies / 32, numberOfBodies / 32);
		//dim3 threadsPerBlock(32, 32); //256 + 128 = 384
		/*int blockSize;
		int minGridSize;
		int gridSize;*/
		/*cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, parallelOverBodies, 0, 0);
		gridSize = (numberOfBodies + blockSize - 1) / blockSize;
		printf("1 Estimated gridSize: %d, blockSize: %d\n", gridSize, blockSize);*/
		parallelOverBodies << < (numberOfBodies + THREADS_PER_BLOCK - 1)/ THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_nbodies, d_activityMap, numberOfBodies, gridLimit, gridDimmension);
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			printf("CUDA error in bodies kernel\n");
		
		// synchronize the device
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			printf("CUDA error synchonizing the device after bodies were simulated\n");
		
		// launch the activity map updater kernel
		/*cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, updateActivityMap, 0, 0);
		gridSize = (numberOfBodies + blockSize - 1) / blockSize;
		printf("2 Estimated gridSize: %d, blockSize: %d\n", gridSize, blockSize);*/
		updateActivityMap << < (numberOfBodies + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_activityMap, numberOfBodies, gridDimmension);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			printf("CUDA error in activity map kernel\n");

		// synchronize the device
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			printf("CUDA error synchonizing the device after activity map was updated\n");

		break;
	}
}
__global__ void updateActivityMap(float * activityMap, int numberOfBodies, int gridDimmension){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	activityMap[idx] /= numberOfBodies;
	activityMap[idx] *= gridDimmension;
}
__global__ void parallelOverBodies(nbodies d_nbodies, float * activityMap, int numberOfBodies, float gridLimit, int gridDimmension){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	/* stores the values euclidian_x, euclidean_y, soft_norm as:
	value.x <- euclidian_x
	value.y <- euclidean_y
	value.z <- soft_norm
	*/
	float3 value; 
	float2 force;
	float2 sum;
	sum.x = 0; sum.y = 0;

	for (int j = 0; j < numberOfBodies; j++){
		// m_j (x_j - x_i) / (|| x_j - x_i ||^2 + softening^2 )^(3/2)
		value.x = d_nbodies.x[j] - d_nbodies.x[idx];
		value.y = d_nbodies.y[j] - d_nbodies.y[idx];
		value.z = (float)pow(value.x * value.x + value.y * value.y + SOFTENING_2, 1.5f);
		// this simation is independent for x or y
		sum.x += (d_nbodies.m[j] * value.x) / value.z;
		sum.y += (d_nbodies.m[j] * value.y) / value.z;
	}
	// Calculate the force
	// F_i = G * m_i * sum
	force.x = G * d_nbodies.m[idx] * sum.x;
	force.y = G * d_nbodies.m[idx] * sum.y;

	// simulate the movement

	// calculate the position
	// WE DO THIS FIRST due to its dependance on current velocity
	// x_t+1 = x_t + dt * v_t
	d_nbodies.x[idx] += dt * d_nbodies.vx[idx];
	d_nbodies.y[idx] += dt * d_nbodies.vy[idx];

	// update the velocity value 
	// acceleration is also computed here, no need for independent computation
	// v_t+1 = v_t + dt * a  // acceleration a_i = F_i / m_i
	d_nbodies.vx[idx] += dt * (force.x / d_nbodies.m[idx]);
	d_nbodies.vy[idx] += dt * (force.y / d_nbodies.m[idx]);

	/*
	compute the position for a body in the activityMap and increase the
	corresponding body count
	index computed according to "The C programming guide" 2nd ed pp.113
	*/
	int col = (int)(d_nbodies.x[idx] / gridLimit);
	int row = (int)(d_nbodies.y[idx] / gridLimit);
	int cell = (int)(gridDimmension * row + col);
	activityMap[cell] += 1.0f;
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
int operation_mode(const int argc, char **argv){
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
int countCommas(const char * buffer){
	unsigned int i, commas = 0;
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
	int i = 0;
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
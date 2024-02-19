#include <stdio.h>
#include <math.h>
#include <stdlib.h>
/// both of these libraries contains the content implementation of cuda.
// #include <cuda.h>
#include <cuda_device_runtime_api.h>
#define bool int
#define true 1
#define false 0

/*
    LEARNING NOTES IN THIS LAB:
    1. We do not have a pass by reference in C, all we can do is to pass by pointer instead.
    2. on doing so, do not forget to wrap your pointer with (*ptr) to be able to access its content
    3. doing *A[i][j] will not affect the result because it try to access the content of A[i][j] not (*A)[i][j]
    4. on passing by pointer, we use & in the calling functions.
    5. on copying elements from or to GPU, we can just send one pointer instead of the total matrix, and the size can be just rows * cols * sizeof(DT)
        instead of creating for loops as we do in the cpu.

*/

/*
    Always we have 6 main steps to follow:
        1. allocate memory on the host.
        2. allocate memory on the device
        3. transfere data to the device.
        4. use the kernel.
        5. transere data back to the host.
        6. free the memory
*/

void freeTheMemoryHost(int ***A, int ***B, int ***Res, int rows)
{
    for (int j = 0; j < rows; j++)
    {
        free((*A)[j]);
        free((*B)[j]);
        free((*Res)[j]);
    }
    free((*A));
    free((*B));
    free((*Res));
}

void freeTheMemoryDevice(int ***A, int ***B, int ***Res, int rows)
{
    cudaFree(A);
    cudaFree(B);
    cudaFree(Res);
}

bool allocateRow(int ***A, int ***B, int ***Res, int rowIdx, int cols_sz)
{
    /// allocate rows for *A
    (*A)[rowIdx] = (int *)malloc(cols_sz);
    if (!(*A)[rowIdx])
    {
        /// free all prevriously allocated rows.
        freeTheMemoryHost(A, B, Res, rowIdx);
        return false;
    }
    return true;
}

bool allocateMemoryOnHost(int ***A, int ***B, int ***Res, int rows, int cols)
{
    int rows_sz = rows * sizeof(int);
    int cols_sz = cols * sizeof(int);
    /// allocate rows.
    (*A) = (int **)malloc(rows_sz);
    if (!(*A))
        return false;

    (*B) = (int **)malloc(rows_sz);
    if (!(*B))
    {
        free((*A));
        return false;
    }

    (*Res) = (int **)malloc(rows_sz);
    if (!(*Res))
    {
        free((*A));
        free((*B));
        return false;
    }

    /// allocate rows.
    for (int i = 0; i < rows; i++)
    {
        /// allocate rows for *A
        if (!allocateRow(A, B, Res, i, cols_sz))
            return false;
        /// allocate rows for *B
        if (!allocateRow(B, A, Res, i, cols_sz))
            return false;

        /// allocate rows for *Res
        if (!allocateRow(Res, B, A, i, cols_sz))
            return false;
    }

    /// fill the matricies
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
        {
            (*A)[i][j] = i * j;
            (*B)[i][j] = i + j;
        }
    return true;
}

void allocateMemoryOnDevice(
    /// GPU matricies
    int **d_A, int **d_B, int **d_C,
    /// rows and cols
    int rows, int cols)
{
    /// allocating memory for rows on the GPU.
    cudaMalloc((void **)&d_A, rows * cols * sizeof(int));
    cudaMalloc((void **)&d_B, rows * cols * sizeof(int));
    cudaMalloc((void **)&d_C, rows * cols * sizeof(int));
}

void transfereDataToTheDevice(
    /// host matricies
    int **h_A, int **h_B,
    /// GPU matricies
    int **d_A, int **d_B,
    /// rows and cols
    int rows, int cols)
{
    cudaMemcpy(d_A, h_A, cols * rows * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, cols * rows * sizeof(int), cudaMemcpyHostToDevice);
}

/*
    without parallelism.
*/
__global__ void kernel0(
    int ***A, int ***B, int ***C, int rows, int cols)
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    (*C)[i][j] = (*A)[i][j] + (*B)[i][j];
}
/*
    each thread produces one output matrix element/
*/
__global__ void kernel1(int ***A, int ***B, int ***C, int rows, int cols)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < rows && j < cols)
        (*C)[i][j] = (*A)[i][j] + (*B)[i][j];
}

/*
    each thread produces one output matrix row/
*/
__global__ void kernel2(int ***A, int ***B, int ***C, int rows, int cols)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < rows)
        return;
    // int j = threadIdx.y + blockIdx.y * blockDim.y;

    for (int j = 0; j < cols; j++)
        // if (i < rows && j < cols)
        (*C)[i][j] = (*A)[i][j] + (*B)[i][j];
}

/*
    each thread produces one output matrix col/
*/
__global__ void kernel3(int ***A, int ***B, int ***C, int rows, int cols)
{
    //    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (j < cols)
        return;
    for (int i = 0; i < rows; i++)
        (*C)[i][j] = (*A)[i][j] + (*B)[i][j];
        // if (i < rows && j < cols)
}

void transfereDataFromTheDevice(
    int **d_C, int **h_C,
    /// rows and cols
    int rows, int cols)
{
    cudaMemcpy(h_C, d_C, cols * rows * sizeof(int), cudaMemcpyDeviceToHost);
}

void printMatrix(int rows, int cols, int **A)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
            printf("%d ", A[i][j]);
        printf("\n");
    }
}

int main()
{
    int **h_A, **h_B, **h_Res;
    int **d_A, **d_B, **d_Res;
    int cols, rows;
    scanf("%d%d", &rows, &cols);

    /// 1. allocate memory on host.
    if (allocateMemoryOnHost(&h_A, &h_B, &h_Res, rows, cols))
    {
        /// 2. allocate memory on device
        allocateMemoryOnDevice(d_A, d_B, d_Res, rows, cols);

        /// 3. transfere data to the device.
        transfereDataToTheDevice(h_A, h_B, d_A, d_B, rows, cols);

        /// 4. use the kernel.
        dim3 blockSize = dim3(1,1,1); 
        dim3 threads = dim3(rows, cols, 1); 
        kernel1<<<>>>();
        kernel2<<<>>>();
        kernel3<<<>>>();

        /// 5. transfere data back to the host.
        transfereDataFromTheDevice(d_Res, h_Res, rows, cols);

        /// 6. free memory
        freeTheMemoryHost(&h_A, &h_B, &h_Res, rows);
    }
    return 0;
}
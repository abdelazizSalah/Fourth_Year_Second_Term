#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define MAX_ERR 1e-6

__global__ void matrix_add(float **out, float **a, float **b, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Handling arbitrary matrix size
    if (row < rows && col < cols) {
        // Req 1
        out[row][col] = a[row][col] + b[row][col];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s input_file output_file\n", argv[0]);
        exit(1);
    }

    FILE *input_file = fopen(argv[1], "r");
    FILE *output_file = fopen(argv[2], "w");
    if (!input_file || !output_file) {
        fprintf(stderr, "Error opening file.\n");
        exit(1);
    }

    int num_test_cases;
    fscanf(input_file, "%d", &num_test_cases);
    
    for (int t = 0; t < num_test_cases; t++) {
        int rows, cols;
        fscanf(input_file, "%d %d", &rows, &cols);

        float **a, **b, **out;
        float **d_a, **d_b, **d_out;

        // Allocate host memory for matrices
        a = (float**)malloc(sizeof(float*) * rows);
        b = (float**)malloc(sizeof(float*) * rows);
        out = (float**)malloc(sizeof(float*) * rows);
        
        for (int i = 0; i < rows; i++) {
            a[i] = (float*)malloc(sizeof(float) * cols);
            b[i] = (float*)malloc(sizeof(float) * cols);
            out[i] = (float*)malloc(sizeof(float) * cols);
        }

        // Read matrix data from file
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fscanf(input_file, "%f", &a[i][j]);
            }
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fscanf(input_file, "%f", &b[i][j]);
            }
        }

        // Allocate device memory for matrices
        cudaMalloc((void***)&d_a, sizeof(float*) * rows);
        cudaMalloc((void***)&d_b, sizeof(float*) * rows);
        cudaMalloc((void***)&d_out, sizeof(float*) * rows);

        float **d_a_data = (float**)malloc(sizeof(float*) * rows);
        float **d_b_data = (float**)malloc(sizeof(float*) * rows);
        float **d_out_data = (float**)malloc(sizeof(float*) * rows);

        for (int i = 0; i < rows; i++) {
            cudaMalloc((void**)&d_a_data[i], sizeof(float) * cols);
            cudaMalloc((void**)&d_b_data[i], sizeof(float) * cols);
            cudaMalloc((void**)&d_out_data[i], sizeof(float) * cols);
        }

        cudaMemcpy(d_a, d_a_data, sizeof(float*) * rows, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, d_b_data, sizeof(float*) * rows, cudaMemcpyHostToDevice);
        cudaMemcpy(d_out, d_out_data, sizeof(float*) * rows, cudaMemcpyHostToDevice);

        // Transfer data from host to device memory
        for (int i = 0; i < rows; i++) {
            cudaMemcpy(d_a_data[i], a[i], sizeof(float) * cols, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b_data[i], b[i], sizeof(float) * cols, cudaMemcpyHostToDevice);
        }

        // Define block size and grid size
        dim3 block_size(16, 16);
        dim3 grid_size((cols + block_size.x - 1) / block_size.x, (rows + block_size.y - 1) / block_size.y);

        // Execute kernel
        matrix_add<<<grid_size, block_size>>>(d_out, d_a, d_b, rows, cols);

        // Transfer data back to host memory
        for (int i = 0; i < rows; i++) {
            cudaMemcpy(out[i], d_out_data[i], sizeof(float) * cols, cudaMemcpyDeviceToHost);
        }

        // Write result to output file
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fprintf(output_file, "%f ", out[i][j]);
            }
            fprintf(output_file, "\n");
        }

        // Free device memory
        for (int i = 0; i < rows; i++) {
            cudaFree(d_a_data[i]);
            cudaFree(d_b_data[i]);
            cudaFree(d_out_data[i]);
        }
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_out);

        free(d_a_data);
        free(d_b_data);
        free(d_out_data);

        // Free host memory
        for (int i = 0; i < rows; i++) {
            free(a[i]);
            free(b[i]);
            free(out[i]);
        }
        free(a);
        free(b);
        free(out);
    }

    fclose(input_file);
    fclose(output_file);

    printf("PASSED\n");

    return 0;
}
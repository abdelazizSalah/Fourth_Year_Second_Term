#include <stdio.h>

void printMatrix(float **mat, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < rows; j++)
            printf("%.2f  ", mat[i][j]);
        printf("\n");
    }
}

int main()
{
    // create file pointer
    FILE *file;

    // try to open the file
    file = fopen("C:\\GitHub\\Fourth_Year_Second_Term\\ParallelComputing\\Labs\\Lab2\\q1TestCase.txt", "r");
    if (!file)
    {
        printf("Error in opening the file");
        return -1;
    }

    // read test cases
    int tc;
    fscanf(file, "%d", &tc);
    printf("%d", tc);

    // read rows, and columns
    int rows, cols;
    fscanf(file, "%d %d", &rows, &cols);
    printf("\n");
    printf("%d %d", rows, cols);

    
    float **a, **b;
    a = (float **)malloc(sizeof(float *) * rows);
    b = (float **)malloc(sizeof(float *) * rows);
    for (int i = 0; i < rows; i++)
    {
        a[i] = (float *)malloc(sizeof(float) * cols);
        b[i] = (float *)malloc(sizeof(float) * cols);
    }

    // read the first matrix.
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            fscanf(file, "%f", &a[i][j]);

    // read the second matrix.
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            fscanf(file, "%f", &b[i][j]);
    printf("\n"); 
    printMatrix(a, rows, cols); 

    printMatrix(b, rows, cols); 

    for (int i = 0; i < rows; i ++) {
        free(a[i]);
        free(b[i]);
    }
    free(a); 
    free(b); 
    fclose(file);

    return 0;
}
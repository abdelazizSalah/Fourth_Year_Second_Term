/*
    @author: Abdelaziz Salah
    @date: 15/2/2024
    @breif: This file contains a C code to solve requirement 1 in parallel computing course.
*/

#include <stdio.h>
#include <stdlib.h>
// #include <cstring>

/*
    @par: matrix: 2d matrix 
    @par: nrows: number of rows
    @par: ncols: number of columns

    @desc: 
        1. initialize a sum variable with 0 to carry the total sum
        2. Iterate over the given number of columns. 
            2.1. Allocate a result string 
            2.2. Initialize it with empty string
            2.3. create another empty string to convert any integer into string in it.
            2.4. iterate from 0 to the given number of rows.
                2.4.1. check if the current number was not negative
                    2.4.1.1. convert the integer into string, and put it inside numInStr
                    2.4.1.2. concatnate it to the result string.
            2.5. convert the result into int, then add it to the sum variable
            2.6. free the allocated string, then repeat.
        3. return the sum. 

*/
int sumOfColumns (int **matrix, int nrows, int ncols) {
    int sum = 0; 
    for (int i = 0 ; i < ncols; i++) {
        char* result = (char *) malloc(sizeof(char) * 200); 
        strcpy(result, "");
        char numInStr[100]; 
        for (int j = 0; j < nrows; j++) {
            if (matrix[j][i] > 0){
                itoa(matrix[j][i], numInStr, 10);
                strcat(result,numInStr); 
            }
        }
        sum += atoi(result);
        free(result);  
    }
    return sum; 
}

void freeMemory (int **matrix, int sz) {
    for (int j = 0; j < sz; j++)
        free(matrix[j]);
    free(matrix);
}

void fillMatrix(int **  matrix, int ncols, int nrows) {
    for (int i = 0; i < nrows; i++)
        for (int j = 0; j < ncols; j++) {
            int num; 
            scanf("%d", &num); 
            matrix[i][j] = num; 
        }
}

int main()
{
    // declaring the number of rows, and columns
    int nrows, ncols;
    scanf("%d %d", &nrows, &ncols);

    // declaring array to hold the inserted data.

    /// 1. define a pointer of pointer -> 2d array
    /// 2. allocate size dynamically using number of rows.
    int **matrix = (int **)malloc(sizeof(int) * nrows);
    if (matrix == NULL) /// fail to allocate memory.
        return -1;

    /// 3. for each row, allocate a new array of size int * cols.
    for (int i = 0; i < nrows; i++)
    {
        /// try to allocate new array
        matrix[i] = (int *) malloc(sizeof(int) * ncols);
        if (matrix[i] == NULL)
        { 
            /// failed to allocate memory
            freeMemory(matrix, i); 
            return -1;
        }
    }

    /// 4. fill the matrix
    fillMatrix(matrix, ncols, nrows); 

    /// 5. call the function
    int sum = sumOfColumns(matrix, nrows, ncols); 
    printf("%d", sum); 

    /// 6. free memory
    freeMemory(matrix, nrows); 
    return 0;
}
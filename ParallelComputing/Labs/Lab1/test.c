#include <stdio.h>
#include <stdlib.h>
int main()
{
    char *str1 = "12";
    char *str2 = "10";
    char *result = malloc (strlen(str1)+strlen(str2) + 20); 
    strcpy(result, str1); 
    // printf("%s", str1 + str2);  // + is not allowed
    printf("%s", strcat(result, str2)); // + is not allowed
    free (result); 
    return 0;
}
#include "stdio.h"
#include <malloc.h>
// #include "omp.h"

void print_array_2d(float *u, int x_size, int y_size);

// Define global variables here
int TIME_ORDER = 2;
int DIMS = 2;

int main(int argc, char const *argv[])
{

    int BORDER_SIZE = 0;
    int SPACE_ORDER = 0;
    int time_m = 0;
    int time_M = 5;
    int GRID_SIZE = 32;
    int x_m = (int)BORDER_SIZE + SPACE_ORDER / 2;
    int x_M = (int)BORDER_SIZE + SPACE_ORDER / 2 + GRID_SIZE;
    int y_m = (int)BORDER_SIZE + SPACE_ORDER / 2;
    int y_M = (int)BORDER_SIZE + SPACE_ORDER / 2 + GRID_SIZE;

    int size_u[] = {GRID_SIZE + 2 * BORDER_SIZE + 2 * SPACE_ORDER, GRID_SIZE + 2 * BORDER_SIZE + 2 * SPACE_ORDER};
    // float **u = (float **)malloc(TIME_ORDER * sizeof(float *));
    // u[0] = (float *)calloc(size_u[0] * size_u[1], sizeof(float));
    // u[1] = (float *)calloc(size_u[0] * size_u[1], sizeof(float));
    float *u = (float *)calloc(size_u[0] * size_u[1], sizeof(float));

    // print_array_2d(u, size_u[0], size_u[1]);

    // GPU working
    #pragma omp target enter data map(to \
                                  : u [0:size_u[0] * (size_u[1] / 2)])
    for (int time = time_m, t0 = (time) % (2), t1 = (time + 1) % (2); time < time_M; time += 1, t0 = (time) % (2), t1 = (time + 1) % (2))
    {
    #pragma omp target teams distribute parallel for collapse(2)
        for (int x = x_m; x < x_M; x += 1)
        {
            for (int y = y_m; y < y_M; y += 1)
            {
                u[y + x * GRID_SIZE] = u[y + x * GRID_SIZE] + 1;
            }
        }
    }
    #pragma omp target update from(u [0:size_u[0] * (size_u[1] / 2)])
    #pragma omp target exit data map(release \
                                 : u [0:size_u[0] * (size_u[1] / 2)])

    // CPU working
    for (int time = time_m, t0 = (time) % (2), t1 = (time + 1) % (2); time < time_M; time += 1, t0 = (time) % (2), t1 = (time + 1) % (2))
    {
        #pragma omp parallel for collapse(2)
        for (int x = x_M / 2; x < x_M; x += 1)
        {
            for (int y = y_m; y < y_M; y += 1)
            {
                u[y + x * GRID_SIZE] = u[y + x * GRID_SIZE] + 1;
            }
        }
    }
    
    print_array_2d(u, size_u[0], size_u[1]);

    free(u);
    printf("\nend.\n");
    return 0;
}

void print_array_2d(float *u, int x_size, int y_size)
{
    // for (int i = 0; i < TIME_ORDER; i++)
    // {
    for (int j = 0; j < y_size; j++)
    {
        for (int k = 0; k < x_size; k++)
        {
            printf("%.f ", u[k + j * x_size]);
        }
        printf("\n");
    }
    printf("\n\n");
    // }
}

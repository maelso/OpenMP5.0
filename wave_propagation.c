#include "stdio.h"
#include <malloc.h>
#include "omp.h"

void print_array_2d(float *u, int x_size, int y_size);

// Define global variables here
int TIME_ORDER = 2;
int DIMS = 2;

int main(int argc, char const *argv[])
{

    int BORDER_SIZE = 0;
    int SPACE_ORDER = 2;
    int time_m = 1;
    int time_M = 10;
    int GRID_SIZE = 141;
    int x_m = (int)BORDER_SIZE + SPACE_ORDER;
    int x_M = (int)BORDER_SIZE + SPACE_ORDER + GRID_SIZE;
    int y_m = (int)BORDER_SIZE + SPACE_ORDER;
    int y_M = (int)BORDER_SIZE + SPACE_ORDER + GRID_SIZE;

    int size_u[] = {GRID_SIZE + 2 * BORDER_SIZE + 2 * SPACE_ORDER, GRID_SIZE + 2 * BORDER_SIZE + 2 * SPACE_ORDER};
    // float *u = (float *)calloc(size_u[0] * size_u[1], sizeof(float));
    float *vp = (float *)calloc(size_u[0] * size_u[1], sizeof(float));
    for(int i=0; i<size_u[0] * size_u[1]; i++)
    {
        vp[i] = 1.5;
    }

    float **u = (float **)malloc(sizeof(float *) * 3);
    u[0] = (float *)calloc(size_u[0] * size_u[1], sizeof(float));
    u[1] = (float *)calloc(size_u[0] * size_u[1], sizeof(float));
    u[2] = (float *)calloc(size_u[0] * size_u[1], sizeof(float));

    u[0][ 72 * size_u[0] + 72] = 1.;
    // print_array_2d(u[0], size_u[0], size_u[1]);
    // printf("\n%f\n\n-------------------------------------------------------------------------------\n", u[0][ 72 * size_u[0] + 72]);

    int num_threads = 1;

    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        // CPU working
        float r1 = 7.840000;
        for (int time = time_m, t0 = (time)%(3), t1 = (time + 1)%(3), t2 = (time + 2)%(3); time <= time_M; time += 1, t0 = (time)%(3), t1 = (time + 1)%(3), t2 = (time + 2)%(3))
        {
            #pragma omp parallel for collapse(2) schedule(guided)
            for (int x = x_m / 2; x < x_M; x += 1)
            {
                for (int y = y_m; y < y_M; y += 1)
                {
                    float r0 = vp[(x*(size_u[0] + 2)) + y + 2] * vp[(x*(size_u[0] + 2)) + y + 2];
                    u[t1][(x*(size_u[0] + 2)) + y + 2] = (float) -3.99999982e-2F*r0*r1*u[t0][(x*(size_u[0] + 2)) + y + 2] + 9.99999955e-3F*(r0*r1*u[t0][(x*(size_u[0] + 1)) + y + 2] + r0*r1*u[t0][(x*(size_u[0] + 2)) + y + 1] + r0*r1*u[t0][(x*(size_u[0] + 2)) + y + 3] + r0*r1*u[t0][(x*(size_u[0] + 3)) + y + 2]) + 1.99999991F * u[t0][(x*(size_u[0] + 2)) + y + 2] - 9.99999955e-1F*u[t2][(x*(size_u[0] + 2)) + y + 2];
                }
            }
        }
        print_array_2d(u[2], size_u[0], size_u[1]);
        // printf("72 => %f \n", u[0][size_u[0]*72 + 72]);
        // printf("72 => %f \n", u[1][size_u[0]*72 + 72]);
        // printf("72 => %f \n", u[2][size_u[0]*72 + 72]);
	
    }
    
    // print_array_2d(u, size_u[0], size_u[1]);

    free(u);
    printf("\nend.\n");
    return 0;
}

void print_array_2d(float *u, int x_size, int y_size)
{
    // for (int i = 0; i < TIME_ORDER; i++)
    // {
    for (int j = 0; j < x_size; j++)
    {
        for (int k = 0; k < y_size; k++)
        {
            printf("%.3f ", u[k + j * x_size]);
        }
        printf("\n");
    }
    printf("\n\n");
    // }
}

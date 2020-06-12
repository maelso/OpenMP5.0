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
    int time_M = 8;
    int GRID_SIZE = 44;
    int x_m = (int)BORDER_SIZE + SPACE_ORDER;
    int x_M = (int)BORDER_SIZE + SPACE_ORDER + GRID_SIZE;
    int y_m = (int)BORDER_SIZE + SPACE_ORDER;
    int y_M = (int)BORDER_SIZE + SPACE_ORDER + GRID_SIZE;

    int size_u[] = {GRID_SIZE + 2 * BORDER_SIZE + 2 * SPACE_ORDER, GRID_SIZE + 2 * BORDER_SIZE + 2 * SPACE_ORDER};
    
    float vp[size_u[0]][size_u[1]];
    float u[TIME_ORDER +1][size_u[0]][size_u[1]];

    //inicializing values
    for (int j = 0; j < size_u[0]; j++)
    {
        for (int k = 0; k < size_u[1]; k++)
        {
            u[0][j][k] = 0.0;
            u[1][j][k] = 0.0;
            u[2][j][k] = 0.0;
	    vp[j][k] = 1.5;
        }
    }
    // source injection
    u[0][22][22] = 1.;

    
    //printing values
    for (int j = 0; j < size_u[0]; j++)
    {
        for (int k = 0; k < size_u[1]; k++)
        {
            printf("%.3f ", u[0][j][k]);
        }
	printf("\n");
    }
    printf("\n\n");

    int num_threads = 8;

    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        // CPU working
        float r1 = 0.0784;
        for (int time = time_m, t0 = (time)%(3), t1 = (time + 1)%(3), t2 = (time + 2)%(3); time <= time_M; time += 1, t0 = (time)%(3), t1 = (time + 1)%(3), t2 = (time + 2)%(3))
        {
            #pragma omp for schedule(guided)
            for (int x = x_m / 2; x < x_M; x += 1)
            {
                for (int y = y_m; y < y_M; y += 1)
                {
                    //float r0 = vp[(x*(size_u[0] + 2)) + y + 2] * vp[(x*(size_u[0] + 2)) + y + 2];
		    float r0 = vp[x + 2][y + 2] * vp[x + 2][y + 2];
        	    u[t1][x + 2][y + 2] = -4.0F*r0*r1*u[t0][x + 2][y + 2] + 1.0F*(r0*r1*u[t0][x + 1][y + 2] + r0*r1*u[t0][x + 2][y + 1] + r0*r1*u[t0][x + 2][y + 3] + r0*r1*u[t0][x + 3][y + 2] - u[t2][x + 2][y + 2]) + 2.0F*u[t0][x + 2][y + 2];
                }
            }
        }
    }
    
    // print_array_2d(u, size_u[0], size_u[1]);
    printf("\n********************************************\n");
    for (int j = 0; j < size_u[0]; j++)
    {
	printf("%d \n", j);
        for (int k = 0; k < size_u[1]; k++)
        {
            printf("%.8f ", u[1][j][k]);
        }
        printf("\n");
    }

    //free(u);
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
            printf("%.7f ", u[k + j * x_size]);
        }
        printf("\n");
    }
    printf("\n\n");
    // }
}

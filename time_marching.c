#define _POSIX_C_SOURCE 200809L

#include "stdio.h"
#include <stdlib.h>
#include "omp.h"

void print_array_2d(float *u, int x_size, int y_size);

// Define global variables here
int TIME_ORDER = 2;
int DIMS = 2;


int main()
{
    int BORDER_SIZE = 0;
    int SPACE_ORDER = 0;
    int time_m = 1;
    int time_M = 10;
    int GRID_SIZE = 8;
    int x_m = (int)BORDER_SIZE + SPACE_ORDER;
    int x_M = (int)BORDER_SIZE + SPACE_ORDER + GRID_SIZE;
    int y_m = (int)BORDER_SIZE + SPACE_ORDER;
    int y_M = (int)BORDER_SIZE + SPACE_ORDER + GRID_SIZE;

    const int size_u[] = {GRID_SIZE + 2 * BORDER_SIZE + 2 * SPACE_ORDER, GRID_SIZE + 2 * BORDER_SIZE + 2 * SPACE_ORDER};

    float (*u)[size_u[0]][size_u[1]];
    posix_memalign((void**)&u, 64, sizeof(float[TIME_ORDER+1][size_u[0]][size_u[1]]));

    //inicializing values
    for (int j = 0; j < size_u[0]; j++)
    {
        for (int k = 0; k < size_u[1]; k++)
        {
            u[0][j][k] = 1.0;
            u[1][j][k] = 1.0;
            u[2][j][k] = 1.0;
        }
    }

    int num_threads = 1;

    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        #pragma omp master
        {
            // #pragma omp target enter data map(to: u[0:1][0:size_u[0]][0:size_u[1]])
            // #pragma omp target enter data map(to: u[1:2][0:size_u[0]][0:size_u[1]])
            // #pragma omp target enter data map(to: u[2:3][0:size_u[0]][0:size_u[1]])
            #pragma omp target enter data map(to: u[0][0:size_u[0]][0:size_u[1]])
            #pragma omp target enter data map(to: u[1][0:size_u[0]][0:size_u[1]])
            // #pragma omp target enter data map(to: u[2][0:size_u[0]][0:size_u[1]])
        }

            for (int time = time_m, t0 = (time)%(2), t1 = (time + 1)%(2); time <= time_M; time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
            {
                #pragma omp master
                {
                    printf("\nT0 = %d\n", t0);
                    for (int x = x_m; x < x_M; x += 1)
                    {
                        for (int y = y_m; y < y_M; y += 1)
                        {
                            printf("%.2f ", u[t0][x][y]);
                        }
                        printf("\n");
                    }
                    #pragma omp target update to( u[0][0:size_u[0]][0:size_u[1]])
                    #pragma omp target update to( u[1][0:size_u[0]][0:size_u[1]])
                    // #pragma omp target update to( u[2][0:size_u[0]][0:size_u[1]])
                    // GPU working
                    #pragma omp target teams distribute parallel for collapse(2) firstprivate(t0, t1) shared(u, x_m, x_M, y_m, y_M, size_u) default(none) 
                    for (int x = x_m; x < x_M; x += 1)
                    {
                        for (int y = y_m; y < y_M; y += 1)
                        {
                            u[t1][x][y] = u[t0][x][y] + 1;
                        }
                    }
                    #pragma omp target update from( u[0][0:size_u[0]][0:size_u[1]])
                    #pragma omp target update from( u[1][0:size_u[0]][0:size_u[1]])
                    // #pragma omp target update from( u[2][0:size_u[0]][0:size_u[1]])
                    printf("\nT1 = %d\n", t1);
                    for (int x = x_m; x < x_M; x += 1)
                    {
                        for (int y = y_m; y < y_M; y += 1)
                        {
                            printf("%.2f ", u[t1][x][y]);
                        }
                        printf("\n");
                    }
                    // printf("\n");
                    // printf("\nT2 = %d\n", t2);
                    // for (int x = x_m; x < x_M; x += 1)
                    // {
                    //     for (int y = y_m; y < y_M; y += 1)
                    //     {
                    //         printf("%.2f ", u[t2][x][y]);
                    //     }
                    //     printf("\n");
                    // }
                    printf("\n-----\n");
                }

            // CPU working
            // float r1 = 7.839999675750732421875;
            // #pragma omp for schedule(guided)
            // for (int x = (x_M - 1) / 2; x < x_M - 1; x += 1)
            // {
            //     for (int y = y_M - 1; y < y_M - 1; y += 1)
            //     {
            //     float r0 = vp[x + 2][y + 2] * vp[x + 2][y + 2];
            //     u[t1][x + 2][y + 2] = -3.99999982e-2F * r0 * r1 * u[t0][x + 2][y + 2] + 
            //                         9.99999955e-3F * (r0 * r1 * u[t0][x + 1][y + 2] + 
            //                                             r0 * r1 * u[t0][x + 2][y + 1] + 
            //                                             r0 * r1 * u[t0][x + 2][y + 3] + 
            //                                             r0 * r1 * u[t0][x + 3][y + 2]) + 
            //                         1.99999991F * u[t0][x + 2][y + 2] - 
            //                         9.99999955e-1F * u[t2][x + 2][y + 2];
            //     }
            // }

            #pragma omp barrier
        }
        #pragma omp barrier
        // #pragma omp target update from( (([3][size_u[0]][size_u[1]]) u) [t1][0:size_u[0]][0:size_u[1]])
        #pragma omp master
        {
            // #pragma omp target update from(u[0:1][0:size_u[0]][0:size_u[1]])
            // #pragma omp target update from(u[1:2][0:size_u[0]][0:size_u[1]])
            // #pragma omp target update from(u[2:3][0:size_u[0]][0:size_u[1]])
            #pragma omp target update from( u[0][0:size_u[0]][0:size_u[1]])
            #pragma omp target update from( u[1][0:size_u[0]][0:size_u[1]])
            // #pragma omp target update from( u[2][0:size_u[0]][0:size_u[1]])

            // #pragma omp target exit data map(release: u[0:1][0:size_u[0]][0:size_u[1]])
            // #pragma omp target exit data map(release: u[1:2][0:size_u[0]][0:size_u[1]])
            // #pragma omp target exit data map(release: u[2:3][0:size_u[0]][0:size_u[1]])
            #pragma omp target exit data map(release: u[0][0:size_u[0]][0:size_u[1]])
            #pragma omp target exit data map(release: u[1][0:size_u[0]][0:size_u[1]])
            // #pragma omp target exit data map(release: u[2][0:size_u[0]][0:size_u[1]])
        }
    }

    printf("***************************************************\n");


    for(int i=0; i<TIME_ORDER+1; i++){
        for(int j=0; j<size_u[0]; j++){
            for(int k=0; k<size_u[1]; k++){
                printf("%.2f ", u[i][j][k]);
            }
            printf("\n");
        }
        printf("\n\n");
    }

    free(u);
    printf("\nend.\n");
    return 0;
}

int get_potition(int size, int x, int y){
    return x + y * size;
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

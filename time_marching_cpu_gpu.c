#define _POSIX_C_SOURCE 200809L

#include "stdio.h"
#include <stdlib.h>
#include "omp.h"

void print_array_2d(float *u, int x_size, int y_size);


int main()
{
    const int TIME_ORDER = 1;
    int BORDER_SIZE = 0;
    int SPACE_ORDER = 0;
    int time_m = 1;
    int time_M = 11;
    int GRID_SIZE = 8;
    int balance_factor = 2;
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
        }
    }

    int num_threads = 8;

    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        #pragma omp master
        {
            // #pragma omp target enter data map(to: u[0:1][0:size_u[0]][0:size_u[1]])
            // #pragma omp target enter data map(to: u[1:2][0:size_u[0]][0:size_u[1]])
            // #pragma omp target enter data map(to: u[2:3][0:size_u[0]][0:size_u[1]])
            #pragma omp target enter data map(to: u[0][0:size_u[0]][0:size_u[1]/balance_factor])
            #pragma omp target enter data map(to: u[1][0:size_u[0]][0:size_u[1]/balance_factor])
            // #pragma omp target enter data map(to: u[2][0:size_u[0]][0:size_u[1]])
        }

            for (int time = time_m, t0 = (time)%(2), t1 = (time + 1)%(2); time <= time_M; time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
            {
                #pragma omp master
                {
                    #pragma omp target update to( u[0][0:size_u[0]][0:size_u[1]/balance_factor])
                    #pragma omp target update to( u[1][0:size_u[0]][0:size_u[1]/balance_factor])
                    // #pragma omp target update to( u[2][0:size_u[0]][0:size_u[1]])
                    // GPU working
                    #pragma omp target teams distribute parallel for collapse(2) firstprivate(t0, t1) shared(u, x_m, x_M, y_m, y_M, balance_factor, size_u) default(none) 
                    for (int x = x_m; x < x_M/balance_factor; x += 1)
                    {
                        for (int y = y_m; y < y_M; y += 1)
                        {
                            u[t1][x][y] = u[t0][x][y] + 1;
                        }
                    }

                    #pragma omp target update from( u[0][0:size_u[0]][0:size_u[1]/balance_factor])
                    #pragma omp target update from( u[1][0:size_u[0]][0:size_u[1]/balance_factor])

                    printf("\n>0\n");
                    for (int x = x_m; x < x_M; x += 1)
                    {
                        for (int y = y_m; y < y_M; y += 1)
                        {
                            printf("%.2f ", u[0][x][y]);
                        }
                        printf("\n");
                    }
                    printf("\n>1 \n");
                    for (int x = x_m; x < x_M; x += 1)
                    {
                        for (int y = y_m; y < y_M; y += 1)
                        {
                            printf("%.2f ", u[1][x][y]);
                        }
                        printf("\n");
                    }
                    printf("\n-----\n");
                }

                // CPU working
                #pragma omp for collapse(2) schedule(guided)
                for (int x = x_M/balance_factor; x < x_M; x += 1)
                {
                    for (int y = y_m; y < y_M; y += 1)
                    {
                        u[t1][x][y] = u[t0][x][y] + 1;
                    }
                }

            #pragma omp barrier
        }
        #pragma omp barrier
        #pragma omp master
        {
            #pragma omp target update from( u[0][0:size_u[0]][0:size_u[1]/balance_factor])
            #pragma omp target update from( u[1][0:size_u[0]][0:size_u[1]/balance_factor])

            #pragma omp target exit data map(release: u[0][0:size_u[0]][0:size_u[1]/balance_factor])
            #pragma omp target exit data map(release: u[1][0:size_u[0]][0:size_u[1]/balance_factor])
        }
    }

    printf("***************************************************\n");


    for(int i=0; i<=TIME_ORDER; i++){
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

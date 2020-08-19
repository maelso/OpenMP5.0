#define _POSIX_C_SOURCE 200809L

#include "stdio.h"
#include <stdlib.h>
#include "omp.h"

void print_array(float *u, int x_size, int y_size);

// Define global variables here
int DEBUG = 0;
int SAVE = 0;
int SAVE_TIME = 1;
int TIME_ORDER = 3; // Change Request: Actually here we have time order 2, the value 3 is representing t0, t1 and t2
int DIMS = 2;


int main()
{
    double start_time, end_time;
    int BORDER_SIZE = 0;
    int SPACE_ORDER = 2;
    int time_m = 1;
    int time_M, GRID_X_SIZE, GRID_Y_SIZE;
    if(DEBUG){
        time_M = 5;
        GRID_X_SIZE = 8;
        GRID_Y_SIZE = 8;
    }else{
        time_M = 6430; // 18 seconds
        // time_M = 100; // 18 seconds
        GRID_X_SIZE = 16384;
        GRID_Y_SIZE = 16384;
    }
    int x_m = (int)BORDER_SIZE + SPACE_ORDER;
    int x_M = (int)BORDER_SIZE + SPACE_ORDER + GRID_X_SIZE;
    int y_m = (int)BORDER_SIZE + SPACE_ORDER;
    int y_M = (int)BORDER_SIZE + SPACE_ORDER + GRID_Y_SIZE;

    const int size_u[] = {GRID_X_SIZE + 2 * BORDER_SIZE + 2 * SPACE_ORDER, GRID_Y_SIZE + 2 * BORDER_SIZE + 2 * SPACE_ORDER};

    float *vp = (float *)calloc(size_u[0] * size_u[1], sizeof(float));
    float *u = (float *)calloc(TIME_ORDER * size_u[0] * size_u[1], sizeof(float));

    int balance_factor = 1;
    int time_offset = size_u[0] * size_u[1];
    
    // time pointers
    float *ut0 = u;
    float *ut1 = u + time_offset;
    float *ut2 = u + 2 * time_offset;
    
    // It will be used to change the time pointers
    float *aux;

    printf("initializing vp...\n");
    //inicializing vp values
    for (int j = 0; j < size_u[0]; j++){
        for (int k = 0; k < size_u[1]; k++){
            vp[j*size_u[0] + k] = 1.5;
        }
    }
    printf("Injecting source...\n");
    // source injection
    ut2[(SPACE_ORDER + GRID_X_SIZE/2) * size_u[0] + (size_u[1]/2)] = 1.;
    if(DEBUG) {
        print_array(ut2, size_u[0], size_u[1]);
    }

    int num_threads = 8;
    printf("Using %d threads\n", num_threads);

    omp_set_num_threads(num_threads);
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        float r1 = 7.839999675750732421875;
        printf("Propagating...\n");
        for (int time = time_m; time <= time_M; time++){
            #pragma omp for collapse(2) schedule(guided) nowait
            for (int x = x_m; x < x_M/balance_factor; x++){
                for (int y = y_m; y < y_M; y++){
                    float r0 = vp[(x * size_u[0]) + y] * vp[(x * size_u[0]) + y];
                    ut1[(x * size_u[0]) + y] =
                            -3.99999982e-2F * r0 * r1 * ut0[(x * size_u[0]) + y] +
                            9.99999955e-3F * (r0 * r1 * ut0[(x * size_u[0] - 1) + y] +
                                              r0 * r1 * ut0[((x-1) * size_u[0]) + y] +
                                              r0 * r1 * ut0[((x+1) * size_u[0]) + y] +
                                              r0 * r1 * ut0[(x * size_u[0] + 1) + y]) +
                            1.99999991F * ut0[(x * size_u[0]) + y] -
                            9.99999955e-1F * ut2[(x * size_u[0]) + y];
                }
            }
            #pragma omp master
            {
                aux = ut1;
                ut1 = ut2;
                ut2 = ut0;
                ut0 = aux;
            }
            #pragma omp barrier
        }
    }
    end_time = omp_get_wtime();
    if(SAVE_TIME){
        FILE* arquivo = fopen("only_cpu.txt", "w");
        if(arquivo == NULL) {
            fprintf(stderr, "Erro ao abrir o arquivo.txt.\n");
            return 1;
        }
        fprintf(arquivo, "Total time = %f seconds\n", end_time-start_time);
        fprintf(arquivo, "\n");
        fclose(arquivo);
    }

    if(SAVE){
        printf("Saving data into .txt file\n");    
        FILE* arquivo = fopen("only_cpu.txt", "w");
        if(arquivo == NULL) {
            fprintf(stderr, "Erro ao abrir o arquivo.txt.\n");
            return 1;
        }
        for(int j=0; j<size_u[0]; j++){
            for(int k=0; k<size_u[1]; k++){
                fprintf(arquivo, "%.20f ", ut0[j*size_u[0] + k] );
            }
            fprintf(arquivo, "\n");
        }
        fclose(arquivo);
        printf("Data saved!\n\n");
    }

    if(DEBUG) {
        print_array(ut0, size_u[0], size_u[1]);
        print_array(ut1, size_u[0], size_u[1]);
        print_array(ut2, size_u[0], size_u[1]);
    }

    free(u);
    free(vp);
    printf("\nend.\n");
    return 0;
}

void print_array(float *u, int x_size, int y_size)
{
    for (int j = 0; j < x_size; j++){
        for (int k = 0; k < y_size; k++){
            printf("%.4f ", u[j*x_size + k]);
        }
        printf("\n");
    }
    printf("\n");
}

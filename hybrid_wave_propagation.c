#define _POSIX_C_SOURCE 200809L

#include "stdio.h"
#include <stdlib.h>
#include "omp.h"

void print_array(float *u, int x_size, int y_size);

// Define global variables here
int DEBUG = 0;
int SAVE = 0;
int TIME_ORDER = 3; // Change Request: Actually here we have time order 2, the value 3 is representing t0, t1 and t2
int DIMS = 2;


int main()
{

    double start_wait, end_wait;
    double elapsed_time = 0;
    int balance_factor = 2;
    int BORDER_SIZE = 0;
    int SPACE_ORDER = 2;
    int time_m = 1;
    int time_M, GRID_X_SIZE, GRID_Y_SIZE;
    if(DEBUG){
        time_M = 5;
        GRID_X_SIZE = 8;
        GRID_Y_SIZE = 8;
    }else{
        time_M = 894;
        GRID_X_SIZE = 1064;
        GRID_Y_SIZE = 1064;
    }
    int x_m = (int)BORDER_SIZE + SPACE_ORDER;
    int x_m_cpu = (int)BORDER_SIZE + SPACE_ORDER + (GRID_X_SIZE/balance_factor);
    int x_M = (int)BORDER_SIZE + SPACE_ORDER + GRID_X_SIZE;
    int x_M_gpu = (int)BORDER_SIZE + SPACE_ORDER + (GRID_X_SIZE/balance_factor);
    int y_m = (int)BORDER_SIZE + SPACE_ORDER;
    int y_M = (int)BORDER_SIZE + SPACE_ORDER + GRID_Y_SIZE;

    const int size_u[] = {GRID_X_SIZE + 2 * BORDER_SIZE + 2 * SPACE_ORDER, GRID_Y_SIZE + 2 * BORDER_SIZE + 2 * SPACE_ORDER};

    float *vp = (float *)calloc(size_u[0] * size_u[1], sizeof(float));
    float *u = (float *)calloc(TIME_ORDER * size_u[0] * size_u[1], sizeof(float));

    int time_offset = size_u[0] * size_u[1];
    int halo = size_u[0] * (SPACE_ORDER/2);
    int gpu_data_domain = ((size_u[0] * size_u[1]) / balance_factor) + halo;
    
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
    #pragma omp parallel private(start_wait, end_wait, elapsed_time)
    {
        #pragma omp master
        {
            #pragma omp target enter data map(to: vp[0:size_u[0] * size_u[1]])
            printf("Sent vp to GPU!\n");
            #pragma omp target enter data map(to: ut0[0:gpu_data_domain])
            printf("Sent ut0 to GPU!\n");
            #pragma omp target enter data map(to: ut1[0:gpu_data_domain])
            printf("Sent ut1 to GPU!\n");
            #pragma omp target enter data map(to: ut2[0:gpu_data_domain])
            printf("Sent ut2 to GPU!\n");
        }
        float r1 = 7.839999675750732421875;
        for (int time = time_m; time <= time_M; time++){
            #pragma omp master
            {
                // GPU working
                #pragma omp target teams distribute parallel for collapse(2) nowait
                for (int x = x_m; x < x_M_gpu; x++){
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
            }
            // CPU working
            #pragma omp for collapse(2) schedule(guided)
            for (int x = x_m_cpu; x < x_M; x++){
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
                // ut1[x * size_u[0] + y] = ut0[x * size_u[0] + y] + 1;
                }
            }
            #pragma omp master
            {
                // #pragma omp taskwait
                #pragma omp target update from(ut1[0:gpu_data_domain-halo])
                aux = ut1;
                ut1 = ut2;
                ut2 = ut0;
                ut0 = aux;
                #pragma omp target update to(ut0[0:gpu_data_domain])
                #pragma omp target update to(ut1[0:gpu_data_domain])
                #pragma omp target update to(ut2[0:gpu_data_domain])

                if(DEBUG) {
                    printf("UT0: %d\n", time);
                    print_array(ut0, size_u[0], size_u[1]);
                }
            }
            start_wait = omp_get_wtime();
            #pragma omp barrier
            end_wait = omp_get_wtime();
            elapsed_time += end_wait - start_wait;
        }
        #pragma omp master
        {
            #pragma omp target exit data map(release: vp[0:size_u[0] * size_u[1]])
            #pragma omp target exit data map(release: ut0[0:gpu_data_domain])
            #pragma omp target exit data map(release: ut1[0:gpu_data_domain])
            #pragma omp target exit data map(release: ut2[0:gpu_data_domain])
        }
    printf("Thread %d waiting time = %f seconds\n", omp_get_thread_num(), elapsed_time);
    }

    if(SAVE){
        printf("Saving data into .txt file...\n");    
        FILE* arquivo = fopen("saida_30_07_20_cpu_gpu.txt", "w");
        if(arquivo == NULL) {
            fprintf(stderr, "Erro ao abrir o arquivo.txt.\n");
            return 1;
        }

        for(int j=0; j<size_u[0]; j++){
            for(int k=0; k<size_u[1]; k++){
                fprintf(arquivo, "%.10f ", ut0[j*size_u[0] + k] );
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
            printf("%.7f ", u[j*x_size + k]);
        }
        printf("\n");
    }
    printf("\n");
}

#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <string.h>

// display_grid
int display_grid(int *grid, int dim) {
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            printf("%d ", grid[i * dim + j]);
        }
        printf("\n");
    }
    printf("\n");
    return 0;
}


// update_cell
int update_cell(int *grid, int *grid_copy, int i, int dim, int iter) {
    int x_ind = i % dim;
    int y_ind = (int)(i / dim);
    int live_neighbor = 0;
    
    //---- counting live neighbors, remember here we should use grid_copy ----
    int x_t, y_t;
    int offset;
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            x_t = (x_ind + i + dim) % dim;
            y_t = (y_ind + j + dim) % dim;
            offset = y_t * dim + x_t;
            live_neighbor += grid_copy[offset];
        }
    }
    live_neighbor -= grid_copy[i];
    
    //-------------------------- update grid[i] -----------------------------
    if ( grid_copy[i] == 1 && (live_neighbor == 2 || live_neighbor == 3)) {
        grid[i] = 1;
    }
    else if (grid_copy[i] == 0 && live_neighbor == 3) {
        grid[i] = 1;
    }
    else {
        grid[i] = 0;
    }

    return 0;
}


/*********************************************************
 * MAIN ENTRY
 *********************************************************/
int main(int argc, char *argv[]) {
    if ( argc != 4 ) {
        printf("usage: ./cell_cpu dimension nSteps frequency\n");
        exit(1);
    }
    int dim = atoi(argv[1]); // dimension
    int nSteps = atoi(argv[2]); // number of steps
    int frequency = atoi(argv[3]); // frequency of printing outputs
    int size = dim * dim; // size of 1-d array
    int *grid = NULL; // grid use to update
    int *grid_copy = NULL; // copy of original grid in each iteration
    clock_t start;
   
    grid = (int *)malloc(size * sizeof(int));
    memset((void *)grid, 0, size);
    grid[1] = 1;
    grid[dim + 2] = 1;
    grid[dim * 2] = 1;
    grid[dim * 2 + 1] = 1;
    grid[dim * 2 + 2] = 1;

    grid_copy = (int *)malloc(size * sizeof(int));
    memcpy((void *)grid_copy, (const void *)grid, size * sizeof(int));
    
    //---- get initial time ----
    start=clock();
    for (int iter = 0; iter < nSteps; iter++) {
        // printf("iteration %d ...\n", iter);
        //------- update grid ------
        for (int i = 0; i < size; i++) {
            update_cell(grid, grid_copy, i, dim, iter);
        }
        //---- update grid_copy ----
        memcpy((void *)grid_copy, (const void *)grid, size * sizeof(int));
        //-------- print out -------
        if(iter % frequency == frequency - 1) {
            printf("Iteration %d / %d: \n", iter + 1, nSteps);
            display_grid(grid, dim);
        }
    }

    //-------- print elapsed time ------
    printf("%f\n", ((float)(clock() - start)) / CLOCKS_PER_SEC);

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

/*********************************************************
 * update_cell
 *********************************************************/
int update_cell(int *grid, int *grid_copy, int i, int dim) {
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
 * display_grid
 *********************************************************/
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

/*********************************************************
 * timeval_subtract
 * Subtract the `struct timeval' values X and Y, storing the result 
 * in RESULT. Return 1 if the difference is negative, otherwise 0.
 *********************************************************/
int timeval_subtract (struct timeval * result, struct timeval * x, struct timeval * y){
    /* Perform the carry for the later subtraction by updating y. */
    if (x->tv_usec < y->tv_usec) {
        int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
        y->tv_usec -= 1000000 * nsec;
        y->tv_sec += nsec;
    }
    if (x->tv_usec - y->tv_usec > 1000000) {
        int nsec = (x->tv_usec - y->tv_usec) / 1000000;
        y->tv_usec += 1000000 * nsec;
        y->tv_sec -= nsec;
    }
    
    /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
    result->tv_sec = x->tv_sec - y->tv_sec;
    result->tv_usec = x->tv_usec - y->tv_usec;
    
    /* Return 1 if result is negative. */
    return x->tv_sec < y->tv_sec;
}

/*********************************************************
 * MAIN ENTRY
 *********************************************************/
int main() {
    int dim = 1600; // dimension
    int nSteps = 64000; // number of steps
    int frequency = 100000; // frequency of printing outputs
    int size = dim * dim; // size of 1-d array
    int *grid = NULL; // grid use to update
    int *grid_copy = NULL; // copy of original grid in each iteration
    struct timeval ta, tb, tresult; // variables for timing
   
    grid = (int *)malloc(size * sizeof(int));
    memset((void *)grid, 0, size);
    grid[1] = 1;
    grid[dim + 2] = 1;
    grid[dim * 2] = 1;
    grid[dim * 2 + 1] = 1;
    grid[dim * 2 + 2] = 1;

    grid_copy = (int *)malloc(size * sizeof(int));
    memcpy((void *)grid_copy, (const void *)grid, size);
    
    //---- get initial time ----
    gettimeofday(&ta, NULL);
    for (int iter = 0; iter < 8; iter++) {
        //------- update grid ------
        for (int i = 0; i < size; i++) {
            update_cell(grid, grid_copy, i, dim);
        }
        //---- update grid_copy ----
        memcpy((void *)grid_copy, (const void *)grid, size);
        //-------- print out -------
        if(iter % frequency == 0) {
            printf("Iteration %d / %d: \n", iter + 1, nSteps);
            display_grid(grid, dim);
        }
    }
    //---- get finishing time ----
    gettimeofday(&tb, NULL);
    
    timeval_subtract(&tresult, &tb, &ta);
    printf("%lu seconds and %lu microseconds.\n", tresult.tv_sec, tresult.tv_usec);

    return 0;
}

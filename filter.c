/*******************************************************************************
*
*  Filter a large array based on the values in a second array.
*
********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <omp.h>
#include <math.h>
#include <assert.h>

/* Example filter sizes */
#define DATA_LEN  512*512*128
#define FILTER_LEN  512
#define NUM_THREADS 4


/* Subtract the `struct timeval' values X and Y,
    storing the result in RESULT.
    Return 1 if the difference is negative, otherwise 0. */
int timeval_subtract (struct timeval * result, struct timeval * x, struct timeval * y)
{
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

/* Function to apply the filter with the filter list in the outside loop */
struct timeval serialFilterFirst ( int data_len, unsigned int* input_array, unsigned int* output_array, int filter_len, unsigned int* filter_list )
{
  /* Variables for timing */
  struct timeval ta, tb, tresult;

  /* get initial time */
  gettimeofday ( &ta, NULL );

  /* for all elements in the filter */ 
  for (int y=0; y<filter_len; y++) { 
    /* for all elements in the data */
    for (int x=0; x<data_len; x++) {
      /* it the data element matches the filter */ 
      if (input_array[x] == filter_list[y]) {
        /* include it in the output */
        output_array[x] = input_array[x];
      }
    }
  }

  /* get initial time */
  gettimeofday ( &tb, NULL );

  timeval_subtract ( &tresult, &tb, &ta );

  printf ("Serial filter first took %lu seconds and %lu microseconds.  Filter length = %d\n", tresult.tv_sec, tresult.tv_usec, filter_len );

  return tresult;
}


/* Function to apply the filter with the filter list in the outside loop */
struct timeval serialDataFirst ( int data_len, unsigned int* input_array, unsigned int* output_array, int filter_len, unsigned int* filter_list )
{
  /* Variables for timing */
  struct timeval ta, tb, tresult;

  /* get initial time */
  gettimeofday ( &ta, NULL );

  /* for all elements in the data */
  for (int x=0; x<data_len; x++) {
    /* for all elements in the filter */ 
    for (int y=0; y<filter_len; y++) { 
      /* it the data element matches the filter */ 
      if (input_array[x] == filter_list[y]) {
        /* include it in the output */
        output_array[x] = input_array[x];
      }
    }
  }

  /* get initial time */
  gettimeofday ( &tb, NULL );

  timeval_subtract ( &tresult, &tb, &ta );

  printf ("Serial data first took %lu seconds and %lu microseconds.  Filter length = %d\n", tresult.tv_sec, tresult.tv_usec, filter_len );

  return tresult;
}

/* Function to apply the filter with the filter list in the outside loop */
struct timeval parallelFilterFirst ( int data_len, unsigned int* input_array, unsigned int* output_array, int filter_len, unsigned int* filter_list )
{
  /* Variables for timing */
  struct timeval ta, tb, tresult;

  /* get initial time */
  gettimeofday ( &ta, NULL );

  /* for all elements in the filter */
  #pragma omp parallel for
  for (int y=0; y<filter_len; y++) { 
    /* for all elements in the data */
    for (int x=0; x<data_len; x++) {
      /* it the data element matches the filter */ 
      if (input_array[x] == filter_list[y]) {
        /* include it in the output */
        output_array[x] = input_array[x];
      }
    }
  }

  /* get initial time */
  gettimeofday ( &tb, NULL );

  timeval_subtract ( &tresult, &tb, &ta );

  printf ("Parallel filter first took %lu seconds and %lu microseconds.  Filter length = %d\n", tresult.tv_sec, tresult.tv_usec, filter_len );

  return tresult;
}


/* Function to apply the filter with the filter list in the outside loop */
struct timeval parallelDataFirst ( int data_len, unsigned int* input_array, unsigned int* output_array, int filter_len, unsigned int* filter_list )
{
  /* Variables for timing */
  struct timeval ta, tb, tresult;

  /* get initial time */
  gettimeofday ( &ta, NULL );

  /* for all elements in the data */
  #pragma omp parallel for 
  for (int x=0; x<data_len; x++) {
    /* for all elements in the filter */
    for (int y=0; y<filter_len; y++) { 
      /* it the data element matches the filter */ 
      if (input_array[x] == filter_list[y]) {
        /* include it in the output */
        output_array[x] = input_array[x];
      }
    }
  }

  /* get initial time */
  gettimeofday ( &tb, NULL );

  timeval_subtract ( &tresult, &tb, &ta );

  printf ("Parallel data first took %lu seconds and %lu microseconds.  Filter length = %d\n", tresult.tv_sec, tresult.tv_usec, filter_len );

  return tresult;
}

struct timeval opt_parallelFilterFirst ( int data_len, unsigned int* input_array, unsigned int* output_array, int filter_len, unsigned int* filter_list )
{
  /* Variables for timing */
  struct timeval ta, tb, tresult;

  /* get initial time */
  gettimeofday ( &ta, NULL );

  /* for all elements in the filter */
  #pragma omp parallel
  for (int y=0; y<filter_len; y++) {
    /* for all elements in the data */
    #pragma omp for nowait
    for (int x=0; x<data_len; x=x+4) {
      /* it the data element matches the filter */ 
      if (input_array[x] == filter_list[y]) {
        output_array[x] = input_array[x];
      }
      if (input_array[x+1] == filter_list[y]) {
        output_array[x+1] = input_array[x+1];
      }
      if (input_array[x+2] == filter_list[y]) {
        output_array[x+2] = input_array[x+2];
      }
      if (input_array[x+3] == filter_list[y]) {
        output_array[x+3] = input_array[x+3];
      }
    }
  }

  /* get initial time */
  gettimeofday ( &tb, NULL );

  timeval_subtract ( &tresult, &tb, &ta );

  printf ("OPT_Parallel filter first took %lu seconds and %lu microseconds.  Filter length = %d\n", tresult.tv_sec, tresult.tv_usec, filter_len );

  return tresult;
}

struct timeval opt_parallelDataFirst ( int data_len, unsigned int* input_array, unsigned int* output_array, int filter_len, unsigned int* filter_list )
{
  /* Variables for timing */
  struct timeval ta, tb, tresult;

  /* get initial time */
  gettimeofday ( &ta, NULL );

  /* for all elements in the data */
  #pragma omp parallel
  for (int x=0; x<data_len; x=x+4) {
    /* for all elements in the filter */
    #pragma omp for nowait
    for (int y=0; y<filter_len; y++) { 
      /* it the data element matches the filter */ 
      if (input_array[x] == filter_list[y]) {
        output_array[x] = input_array[x];
      }
      if (input_array[x+1] == filter_list[y]) {
        output_array[x+1] = input_array[x+1];
      }
      if (input_array[x+2] == filter_list[y]) {
        output_array[x+2] = input_array[x+2];
      }
      if (input_array[x+3] == filter_list[y]) {
        output_array[x+3] = input_array[x+3];
      }
    }
  }

  /* get initial time */
  gettimeofday ( &tb, NULL );

  timeval_subtract ( &tresult, &tb, &ta );

  printf ("OPT_Parallel data first took %lu seconds and %lu microseconds.  Filter length = %d\n", tresult.tv_sec, tresult.tv_usec, filter_len );

  return tresult;
}

void checkData ( unsigned int * serialarray, unsigned int * parallelarray )
{
  for (int i=0; i<DATA_LEN; i++)
  {
    if (serialarray[i] != parallelarray[i])
    {
      printf("Data check failed offset %d\n", i );
      return;
    }
  }
}


int main( int argc, char** argv )
{
  /* loop variables */
  int x,y;

  /* Create matrixes */
  unsigned int * input_array;
  unsigned int * serial_array;
  unsigned int * output_array;
  unsigned int * filter_list;

  /* Initialize the data. Values don't matter much. */
  posix_memalign ( (void**)&input_array, 4096,  DATA_LEN * sizeof(unsigned int));
//  input_array = (unsigned int*) posix_memalign ( DATA_LEN * sizeof(unsigned int), 4096);
  for (x=0; x<DATA_LEN; x++)
  {
    input_array[x] = x % 2048;
  }

  /* start with an empty *all zeros* output array */
  posix_memalign ( (void**)&output_array, 4096, DATA_LEN * sizeof(unsigned int));
  memset ( output_array, 0, DATA_LEN );
  posix_memalign ( (void**)&serial_array, 4096, DATA_LEN * sizeof(unsigned int));
  memset ( serial_array, 0, DATA_LEN );

  /* Initialize the filter. Values don't matter much. */
  filter_list = (unsigned int*) malloc ( FILTER_LEN * sizeof(unsigned int));
  for (y=0; y<FILTER_LEN; y++){
    filter_list[y] = y;
  }

  while(1){
  printf("\n\n");
  printf("**************************************************************************\n");
  printf("**  If you want to run code for loop efficiency, please enter 1;        **\n");
  printf("**  If you want to run code for paralle single time, please enter 2;    **\n");
  printf("**  If you want to run code for speedup, please enter 3;                **\n");
  printf("**  If you want to run code for scaleup, please enter 4;                **\n");
  printf("**  If you want to run optimized code, please enter 5;                  **\n");
  printf("**  If you want to exit, please enter 0;                                **\n");
  printf("**************************************************************************\n");
  printf("NOTICE: for scaleup, you need to change the #DATA_LEN and #NUM_THREADS manually in the source file, and compile and run it again\n");

  char c = getchar();

  if(c == '1'){
    FILE *f_data = fopen( "1_dataFirst.txt", "at+");
    FILE *f_filter = fopen( "1_filterFirst.txt", "at+");
    struct timeval time_record;
    int sec, usec;
    for ( int filter_len =1; filter_len<=1024; filter_len*=2) {
      time_record = serialDataFirst ( DATA_LEN, input_array, serial_array, filter_len, filter_list );
      sec = time_record.tv_sec;
      usec = time_record.tv_usec;
      //printf("%d %d \n", sec, usec);
      fprintf(f_data, "%d %d\n", sec, usec);
      memset ( output_array, 0, DATA_LEN );

      time_record = serialFilterFirst ( DATA_LEN, input_array, output_array, filter_len, filter_list );
      sec = time_record.tv_sec;
      usec = time_record.tv_usec;
      checkData ( serial_array, output_array );
      fprintf(f_filter, "%d %d\n", sec, usec);
      memset ( output_array, 0, DATA_LEN );
    }
    fprintf(f_data, "\n");
    fprintf(f_filter, "\n");
    fclose(f_data);
    fclose(f_filter);
  }

  else if( c == '2' ){
    FILE *f_parallel = fopen( "2_parallel.txt", "at+");
    struct timeval time_record;
    int sec, usec;

    time_record = serialDataFirst ( DATA_LEN, input_array, serial_array, FILTER_LEN, filter_list );
    sec = time_record.tv_sec;
    usec = time_record.tv_usec;
    // for(int j = 0; j<=5; j++){
    //     printf("%d ", serial_array[j]);
    //   }
    //   printf("\n");
    fprintf(f_parallel, "%d %d\n", sec, usec);
    memset ( output_array, 0, DATA_LEN );

    time_record = parallelFilterFirst ( DATA_LEN, input_array, output_array, FILTER_LEN, filter_list );
    sec = time_record.tv_sec;
    usec = time_record.tv_usec;
    checkData ( serial_array, output_array );
    fprintf(f_parallel, "%d %d\n", sec, usec);
    memset ( output_array, 0, DATA_LEN );

    time_record = parallelDataFirst ( DATA_LEN, input_array, output_array, FILTER_LEN, filter_list );
    sec = time_record.tv_sec;
    usec = time_record.tv_usec;
    checkData ( serial_array, output_array );
    fprintf(f_parallel, "%d %d\n", sec, usec);
    memset ( output_array, 0, DATA_LEN );

    fprintf(f_parallel, "\n");
    fclose(f_parallel);
  }

  else if(c == '3'){
    serialDataFirst ( DATA_LEN, input_array, serial_array, FILTER_LEN, filter_list );
    memset ( output_array, 0, DATA_LEN );

    FILE *f_data = fopen( "3_speedup_datafirst.txt", "at+");
    FILE *f_filter = fopen( "3_speedup_filterfirst.txt", "at+");
    struct timeval time_record;
    int sec, usec;
    for(int i=1; i<=NUM_THREADS; i=i*2){
      omp_set_num_threads(i);
      printf("Thread number: %d\n", i);

      time_record = parallelFilterFirst ( DATA_LEN, input_array, output_array, FILTER_LEN, filter_list );
      sec = time_record.tv_sec;
      usec = time_record.tv_usec;
      checkData ( serial_array, output_array );
      fprintf(f_filter, "%d %d\n", sec, usec);
      memset ( output_array, 0, DATA_LEN );

      time_record = parallelDataFirst ( DATA_LEN, input_array, output_array, FILTER_LEN, filter_list );
      sec = time_record.tv_sec;
      usec = time_record.tv_usec;
      checkData ( serial_array, output_array );
      fprintf(f_data, "%d %d\n", sec, usec);
      memset ( output_array, 0, DATA_LEN );
    }

    fprintf(f_data, "\n");
    fprintf(f_filter, "\n");
    fclose(f_data);
    fclose(f_filter);
  }

  else if(c == '4'){
        FILE *f_d = fopen( "4_scaleup_datafirst_1.txt", "at+");
    FILE *f_f = fopen( "4_scaleup_filterfirst_1.txt", "at+");
    struct timeval time_record;
    int sec, usec;

    time_record = serialDataFirst ( DATA_LEN, input_array, serial_array, FILTER_LEN, filter_list );
    sec = time_record.tv_sec;
    usec = time_record.tv_usec;

    omp_set_num_threads(NUM_THREADS);
    time_record = parallelFilterFirst ( DATA_LEN, input_array, output_array, FILTER_LEN, filter_list );
    sec = time_record.tv_sec;
    usec = time_record.tv_usec;
    checkData ( serial_array, output_array );
    fprintf(f_f, "%d %d\n", sec, usec);
    memset ( output_array, 0, DATA_LEN );

    time_record = parallelDataFirst ( DATA_LEN, input_array, output_array, FILTER_LEN, filter_list );
    sec = time_record.tv_sec;
    usec = time_record.tv_usec;
    checkData ( serial_array, output_array );
    fprintf(f_d, "%d %d\n", sec, usec);
    memset ( output_array, 0, DATA_LEN );
  }

  else if(c == '5'){
    serialDataFirst ( DATA_LEN, input_array, serial_array, FILTER_LEN, filter_list );

    parallelDataFirst ( DATA_LEN, input_array, output_array, FILTER_LEN, filter_list );
    checkData ( serial_array, output_array );
    memset ( output_array, 0, DATA_LEN );

    opt_parallelDataFirst ( DATA_LEN, input_array, output_array, FILTER_LEN, filter_list );
    checkData ( serial_array, output_array );
    memset ( output_array, 0, DATA_LEN );

    opt_parallelFilterFirst ( DATA_LEN, input_array, output_array, FILTER_LEN, filter_list );
    checkData ( serial_array, output_array );
    memset ( output_array, 0, DATA_LEN );
  }

  else if(c == '0'){
    break;
  }


  }
}
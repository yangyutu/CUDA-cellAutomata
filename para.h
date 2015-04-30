
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

template< typename T >
void swap( T& a, T& b ) {
    T t = a;
    a = b;
    b = t;
}


struct DataBlock
{
int *outbitmap;
int *dev_in;
int *dev_out;
int *bitmap;
};


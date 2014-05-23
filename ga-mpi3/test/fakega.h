#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <mpi.h>

typedef struct _GA { 
	MPI_Win      ga_win; 
	/* Datatype and size */ 
	MPI_Datatype dtype;              
	int          dtype_size; 
	/* sizes of the global array */ 
	int          dim1, dim2, chunk2; 
	MPI_Comm	comm;
	/* window base */
	void * win_base;
} *GA;

typedef enum transfer_type_e_ 
{
	PUT	= 1,
	GET   	= 2,
	ACC	= 4
} transfer_type; 

extern int nprocs, proc;

int _ga_create( MPI_Comm comm, int dim1, int dim2, MPI_Datatype dtype, GA * ga ); 

int _ga_transfer (GA ga, int ilo, int jlo, int ihigh, int jhigh, void * buf, int * ld, transfer_type xtype); 

int _ga_duplicate( GA ga, GA * dup );

int _ga_destroy( GA ga ); 

int _nga_put( GA ga, int * lo, int * hi, void * buf, int * ld );

int _nga_get( GA ga, int * lo, int * hi, void * buf, int * ld ); 

int _nga_acc( GA ga, int * lo, int * hi, void * buf, int * ld ); 

long _ga_read_inc( GA ga, int * subscript, long inc );

void _ga_error (char * str, int code);

int _ga_zero ( GA ga );

void _ga_sync (void);

void _ga_flush (GA ga); 

double _ga_wtime (void);

void _ga_add(double alpha, GA g_a, double beta, GA g_b, GA g_c);

void _ga_transpose( GA g_a, GA g_b);

void _ga_dgemm(char transa[1], char transb[1], int m, int n, int k, double alpha, GA g_a, GA g_b, double beta, GA g_c );

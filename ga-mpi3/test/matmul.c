/* Sample matrix-matrix multiplication */

#include "fakega.h"

#ifdef USE_GSL_CBLAS
#include <gsl/gsl_cblas.h>
#else
#include <cblas.h>
#endif
#include <time.h>

#define   NDIMS   2
#define   MAX_DTYPE_LEN	50

int nprocs, proc;

/* Square matrix-matrix multiplication */
void matrix_multiply(int M, int N, int K, 
		int blockX, int blockY) 
{
	/* Local buffers and Global arrays declaration */
	double *a, *b, *c, *atrans;
	int dims[NDIMS], lo[NDIMS], hi[NDIMS], ld[NDIMS]; /* dim of blocks */
	GA g_a, g_b, g_c, g_cnt;

	double alpha = 1.0, beta=1.0;
	long count_p = 0, next_p = 0;
	long count_gac = 0, next_gac = 0;
	double t1,t2,seconds;
	int status;

	char type_name[MAX_DTYPE_LEN];
	int resultlen = 50;
	int * rdcnt;


	if ((M % blockX) != 0 || (M % blockY) != 0 || (N % blockX) != 0 || (N % blockY) != 0 
			|| (K % blockX) != 0 || (K % blockY) != 0)
		_ga_error("Dimension size M/N/K is not divisible by X/Y block sizes", 101);

	if ((blockX != blockY) && (blockX * blockY) <= 0)
		_ga_error("Square blocks greater than 0 expected",102);
	
	/* Allocate/Set process local buffers */
	MPI_Alloc_mem(blockX * blockY * sizeof(double), MPI_INFO_NULL, &a);
	MPI_Alloc_mem(blockX * blockY * sizeof(double), MPI_INFO_NULL, &atrans);
	MPI_Alloc_mem(blockX * blockY * sizeof(double), MPI_INFO_NULL, &b);
	MPI_Alloc_mem(blockX * blockY * sizeof(double), MPI_INFO_NULL, &c);
	/* Configure array dimensions...considering square arrays only */
	for(int i = 0; i < NDIMS; i++) {
		dims[i]  = N;
		ld[i] = blockX;
	}

	/* create a global array g_a and duplicate it to get g_b and g_c*/
	status = _ga_create(MPI_COMM_WORLD, dims[0], dims[1], MPI_DOUBLE, &g_a); 
	if (!status) 
		_ga_error("NGA_Create failed: A", NDIMS);

#if DEBUG>1
	if (proc == 0) { 
		MPI_Type_get_name (g_a->dtype, type_name, &resultlen);
		printf("Created Global array A with datatype - %s\n", type_name);
	}
#endif
	/* Ditto for C and B */
	_ga_duplicate(g_a, &g_b);
	_ga_duplicate(g_a, &g_c);

	if (!g_b || !g_c) 
		_ga_error("GA_Duplicate failed",NDIMS);

#if DEBUG>1
	if (proc == 0) { 
		MPI_Type_get_name (g_b->dtype, type_name, &resultlen);
		printf("Created Global array B with datatype - %s\n", type_name);
		MPI_Type_get_name (g_c->dtype, type_name, &resultlen);
		printf("Created Global array C with datatype - %s\n", type_name);
	}
#endif
	/* Subscript array for read-incr */
	rdcnt = malloc (NDIMS * sizeof(int));
	memset (rdcnt, 0, NDIMS * sizeof(int));
	/* Create global array for nxtval */	
	status = _ga_create(MPI_COMM_WORLD, 1, 1, MPI_LONG, &g_cnt); 

	if (!g_cnt) 
		_ga_error("Shared counter failed",1);

	_ga_zero(g_cnt);
#if DEBUG>1	
	if (proc == 0) {
		MPI_Type_get_name (g_cnt->dtype, type_name, &resultlen);
		printf("Created Global array g_cnt (counter) - %s\n", type_name);
	}
#endif
#if DEBUG>1	
	/* initialize data in matrices a and b */
	if(proc == 0)
		printf("Initialized counters to 0\n");
#endif
	/* Populate block arrays with dummy data */
	int w = 0; 
	int l = 7;
	for(int i = 0; i < blockX; i++) {
		for(int j = 0; j < blockY; j++) {
			a[i*blockY + j] = (double)(++w%29);
			b[i*blockY + j] = (double)(++l%37);
		}
	}

#if DEBUG>1	
	if(proc == 0)
		printf("Initialized local buffers - a and b in rank #0 only\n");
#endif
	/* Copy data to global arrays g_a and g_b from local buffers */
	/* Initialize read_cnt */
	next_p = _ga_read_inc(g_cnt, rdcnt, (long)1);
	for (int i = 0; i < N; i+=blockX) {
#if DEBUG>1
		printf ("%d: next_p = %ld and count_p = %ld\n",proc,next_p,count_p);
#endif	
		if (next_p == count_p) {
			for (int j = 0; j < N; j+=blockY) {
				/* Indices of patch */
				lo[0] = i; lo[1] = j;
				hi[0] = lo[0] + blockX; hi[1] = lo[1] + blockY;
				hi[0] = hi[0]-1; hi[1] = hi[1]-1;

				_nga_put(g_a, lo, hi, a, ld); 
				_nga_put(g_b, lo, hi, b, ld);
#if DEBUG>1
				printf ("%d: PUT_GA_A_B: lo[0,1] = %d,%d and hi[0,1] = %d,%d\n",proc,lo[0],lo[1],hi[0],hi[1]);
#endif
			}
			next_p = _ga_read_inc(g_cnt, rdcnt, (long)1);
		}	
		count_p++;
	}
	/* Synchronize all processors to make sure puts from 
	   nprocs has finished before proceeding with dgemm */
	_ga_sync();
	_ga_zero(g_cnt);

#if DEBUG>1
	if (proc == 0)
		printf ("\nAfter initial NGA_PUT to global arrays - A and B\n\n\n");
#endif

	t1 = _ga_wtime();
	
	next_gac = _ga_read_inc(g_cnt, rdcnt, (long)1);
	for (int m = 0; m < M; m+=blockX) {
		for (int n = 0; n < N; n+=blockY) {
			memset (c, 0, sizeof(double) * blockX * blockY);
#if DEBUG>1
			printf ("%d: next_gac = %ld and count_gac = %ld\n",proc,next_gac,count_gac);
#endif
			if (next_gac == count_gac)	
			{
				for (int k = 0; k < K; k+=blockX)
				{
					/* A = m x k */
					lo[0] = m; lo[1] = k;
					hi[0] = blockX + lo[0]; hi[1] = blockY + lo[1];
					hi[0] = hi[0]-1; hi[1] = hi[1]-1;
#if DEBUG>1
					printf ("%d: GET GA_A: lo[0,1] = %d,%d and hi[0,1] = %d,%d\n",proc,lo[0],lo[1],hi[0],hi[1]);
#endif
					_nga_get(g_a, lo, hi, a, ld);
					/* Perform A^T and store in atrans */
					for (int i=0; i< hi[0] - lo[0]+1; i++)
						for (int j=0; j< hi[0] - lo[0]+1; j++)
							atrans[j*blockX+i] = a[i*blockY+j];
					/* A = A + A^T */
					for (int i=0; i< hi[0] - lo[0]+1; i++)
						for (int j=0; j< hi[0] - lo[0]+1; j++)
							a[i*blockY+j] += atrans[i*blockX+j];                          
					/* B = k x n */
					lo[0] = k; lo[1] = n;
					hi[0] = blockX + lo[0]; hi[1] = blockY + lo[1];				
					hi[0] = hi[0]-1; hi[1] = hi[1]-1;
#if DEBUG>1
					printf ("%d: GET_GA_B: lo[0,1] = %d,%d and hi[0,1] = %d,%d\n",proc,lo[0],lo[1],hi[0],hi[1]);
#endif
					_nga_get(g_b, lo, hi, b, ld); 

					/* TODO I am assuming square matrix blocks, further testing/work 
					   required for rectangular matrices */
					cblas_dgemm ( CblasRowMajor, CblasNoTrans, /* TransA */CblasNoTrans, /* TransB */
							blockX /* M */, blockY /* N */, blockX /* K */, alpha,
							a, blockX, /* lda */ b, blockY, /* ldb */
							beta, c, blockX /* ldc */);
				} /* END LOOP K */
				/* C = m x n */
				lo[0] = m; lo[1] = n;
				hi[0] = blockX + lo[0]; hi[1] = blockY + lo[1];				
				hi[0] = hi[0]-1; hi[1] = hi[1]-1;
#if DEBUG>1
				printf ("%d: ACC_GA_C: lo[0,1] = %d,%d and hi[0,1] = %d,%d\n",proc,lo[0],lo[1],hi[0],hi[1]);
#endif
				_nga_acc(g_c, lo, hi, c, ld); 
				next_gac = _ga_read_inc(g_cnt, rdcnt, (long)1);
			} /* ENDIF if count == next */
			count_gac++;
		} /* END LOOP N */
	} /* END LOOP M */
	_ga_sync();

	t2 = _ga_wtime();
	seconds = t2 - t1;
	double total_secs;

	MPI_Reduce(&seconds, &total_secs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (proc == 0)
		printf("Time taken for MM (secs):%lf \n", (total_secs/nprocs));

	/* Clear local buffers */
	MPI_Free_mem(a);
	MPI_Free_mem(atrans);
	MPI_Free_mem(b);
	MPI_Free_mem(c);
	/* Deallocate arrays */
	_ga_destroy(g_a);
	_ga_destroy(g_b);
	_ga_destroy(g_c);
	_ga_destroy(g_cnt);

}

/*************TESTING****************/
/*************TESTING****************/
/*************TESTING****************/
#if 0
#define TOLERANCE 0.1
void verify(int g_a, int g_b, int g_c, int *lo, int *hi, int *ld, int N) 
{

	double rchk, alpha=1.0, beta=0.0, temp_beta=1.0;
	int g_chk, g_atrans;
	int me = proc;

	g_chk = _ga_duplicate(g_a);
	g_atrans = _ga_duplicate(g_a);
	if(!g_chk || !g_atrans) _ga_error("duplicate failed",NDIMS);
	_ga_sync();

	_ga_transpose(g_a,g_atrans);
	_ga_add(&alpha,g_a,&temp_beta,g_atrans,g_a);
	_ga_dgemm('n', 'n', N, N, N, 1.0, g_a,
			g_b, 0.0, g_chk);

	_ga_sync();

	alpha=1.0, beta=-1.0;
	_ga_add(&alpha, g_c, &beta, g_chk, g_chk);
	rchk = _ga_ddot(g_chk, g_chk);

	if (me==0) {
		printf("Normed difference in matrices: %12.4e\n", rchk);
		if(rchk < -TOLERANCE || rchk > TOLERANCE)
			GA_Error("Matrix multiply verify failed",0);
		else
			printf("Matrix Mutiply OK\n");
	}

	_ga_destroy(g_chk);
	_ga_destroy(g_atrans);
}
/*************TESTING****************/
/*************TESTING****************/
/*************TESTING****************/
#endif
int main(int argc, char **argv) 
{
	int M, N, K; /* */
	int blockX, blockY;

	if (argc == 6) {
		M = atoi(argv[1]);
		N = atoi(argv[2]);
		K = atoi(argv[3]);
		blockX = atoi(argv[4]);
		blockY = atoi(argv[5]);
	}
	else {
		printf("Please enter ./a.out <M> <N> <K> <BLOCK-X-LEN> <BLOCK-Y-LEN>");
		exit(-1);
	}

	MPI_Init(&argc, &argv);

	MPI_Comm_rank (MPI_COMM_WORLD, &proc);
	MPI_Comm_size (MPI_COMM_WORLD, &nprocs);

	if(proc == 0) {
		printf("Using %d processes\n", nprocs); 
		printf("\nSize of M, N, K: %d - %d - %d & size of BLOCK: %d X %d \n", M, N, K, blockX, blockY);
		printf("\n**********************************************************\n");
		fflush(stdout);
	}

	matrix_multiply(M, N, K, blockX, blockY);

	if(proc == 0)
		printf("\nTerminating ..\n");

	MPI_Finalize();    

	return 0;
}

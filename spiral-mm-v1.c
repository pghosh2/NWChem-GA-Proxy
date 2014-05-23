/* Sample matrix-matrix multiplication */

#include "ga.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "macdecls.h"
#include <mpi.h>
#include <armci.h>

//#include "/usr/include/gsl/gsl_cblas.h"
#include <cblas.h>

#define   NDIMS   2

/* Naive dgemm */
/* void _my_dgemm_ (double *a, int lda, double *b, int ldb, double *c, int ldc, int M, int N, int K, double alpha, double beta) */
void _my_dgemm_ (double *a, int lda, double *b, int ldb, 
		double *c, int ldc, int M, int N, int K, 
		double alpha, double beta)
{
	for(int i = 0; i < M; i++) 
	{
		for(int j = 0; j < N; j++) 
		{
			c[i*ldc + j] = beta * c[i*ldc + j];

			for(int k = 0; k < K; k++) 
			{
				c[i*ldc + j] += alpha * (a[i*lda + k] * b[k*ldb + j]);
			}
		} 
	}
}

/* Correctness test */
void verify(int g_a, int g_b, int g_c, int *lo, int *hi, int *ld, int N);

/* Square matrix-matrix multiplication */
void matrix_multiply(int M, int N, int K, 
		int blockX_len, int blockY_len) 
{
	/* Local buffers and Global arrays declaration */
	double *a=NULL, *b=NULL, *c[2]={NULL,NULL}, *atrans=NULL;

	int dims[NDIMS], ld[NDIMS], chunks[NDIMS];
	int lo[NDIMS], hi[NDIMS], cdims[NDIMS]; /* dim of blocks */

	int g_a, g_b, g_c, g_cnt, g_cnt2;
	int offset;
	double alpha = 1.0, beta=0.0;
	int count_p = 0, next_p = 0;
	int count_gac = 0, next_gac = 0;
	double t1,t2,seconds;
        ga_nbhdl_t nbh[2];
        int count_acc = 0;
        int i,j;

	/* Find local processor ID and the number of processes */
	int proc=GA_Nodeid(), nprocs=GA_Nnodes();

	if ((M % blockX_len) != 0 || (M % blockY_len) != 0 || (N % blockX_len) != 0 || (N % blockY_len) != 0 
			|| (K % blockX_len) != 0 || (K % blockY_len) != 0)
		GA_Error("Dimension size M/N/K is not divisible by X/Y block sizes", 101);

	/* Allocate/Set process local buffers */
	a = ARMCI_Malloc_local (blockX_len * blockY_len * sizeof(double)); 
	atrans = ARMCI_Malloc_local (blockX_len * blockY_len * sizeof(double)); 
	b = ARMCI_Malloc_local (blockX_len * blockY_len * sizeof(double)); 
	c[0] = ARMCI_Malloc_local (blockX_len * blockY_len * sizeof(double));
	c[1] = ARMCI_Malloc_local (blockX_len * blockY_len * sizeof(double));

	cdims[0] = blockX_len;
	cdims[1] = blockY_len;	

	/* Configure array dimensions */
	for(int i = 0; i < NDIMS; i++) {
		dims[i]  = N;
		chunks[i] = cdims[i];
		ld[i]    = cdims[i]; /* leading dimension/stride of the local buffer */
	}

	/* create a global array g_a and duplicate it to get g_b and g_c*/
	//g_a = NGA_Create(C_DBL, NDIMS, dims, "array A", chunks);
	g_a = GA_Create_handle();
        GA_Set_array_name(g_a, "Array A");
        GA_Set_data(g_a, NDIMS, dims, C_DBL);
        GA_Set_chunk(g_a, cdims);
        GA_Allocate(g_a);

	if (!g_a) 
		GA_Error("NGA_Create failed: A", NDIMS);

#if DEBUG>1
	if (proc == 0) 
		printf("  Created Array A\n");
#endif
	/* Ditto for C and B */
	g_b = GA_Duplicate(g_a, "array B");
	g_c = GA_Duplicate(g_a, "array C");

	if (!g_b || !g_c) 
		GA_Error("GA_Duplicate failed",NDIMS);
	if (proc == 0) 
		printf("Created Arrays B and C\n");

	/* Subscript array for read-incr */
        int rdcnt[1];
        rdcnt[0] = 0;

	/* Create global array of nprocs elements for nxtval */	
	int counter_dim[1];
	counter_dim[0] = nprocs;

	g_cnt = NGA_Create(C_INT, 1, counter_dim, "Shared counter", NULL);

	if (!g_cnt) 
		GA_Error("Shared counter failed",1);

	g_cnt2 = GA_Duplicate(g_cnt, "another shared counter");

	if (!g_cnt2) 
		GA_Error("Another shared counter failed",1);

	GA_Zero(g_cnt);
	GA_Zero(g_cnt2);

        /* Print GA distribution */
        //GA_Print_distribution(g_a);
        //GA_Print_distribution(g_b);
        //GA_Print_distribution(g_c);

#if DEBUG>1	
	/* initialize data in matrices a and b */
	if(proc == 0)
		printf("Initializing local buffers - a and b\n");
#endif
	int w = 0; 
	int l = 7;
	for(int i = 0; i < cdims[0]; i++) {
		for(int j = 0; j < cdims[1]; j++) {
			a[i*cdims[1] + j] = (double)(++w%29);
			b[i*cdims[1] + j] = (double)(++l%37);
		}
	}

	/* Copy data to global arrays g_a and g_b from local buffers */
	next_p = NGA_Read_inc(g_cnt2,rdcnt,(long)1);
	for (int i = 0; i < N; i+=cdims[0]) 
	{
		if (next_p == count_p) {
			for (int j = 0; j < N; j+=cdims[1])
			{
				/* Indices of patch */
				lo[0] = i;
				lo[1] = j;
				hi[0] = lo[0] + cdims[0];
				hi[1] = lo[1] + cdims[1];

				hi[0] = hi[0]-1;
				hi[1] = hi[1]-1;
#if DEBUG>1
				printf ("%d: PUT_GA_A_B: lo[0,1] = %d,%d and hi[0,1] = %d,%d\n",proc,lo[0],lo[1],hi[0],hi[1]);
#endif
				NGA_Put(g_a, lo, hi, a, ld);
				NGA_Put(g_b, lo, hi, b, ld);

			}
			next_p = NGA_Read_inc(g_cnt2,rdcnt,(long)1);
		}		
		count_p++;
	}


#if DEBUG>1
	printf ("After NGA_PUT to global - A and B arrays\n");
#endif
	/* Synchronize all processors to make sure puts from 
	   nprocs has finished before proceeding with dgemm */
	GA_Sync();

	t1 = GA_Wtime();

	next_gac = NGA_Read_inc(g_cnt,rdcnt,(long)1);
	for (int m = 0; m < N; m+=cdims[0])
	{
		for (int k = 0; k < N; k+=cdims[0])
		{
			if (next_gac == count_gac)	
			{
				/* A = m x k */
				lo[0] = m; lo[1] = k;
				hi[0] = cdims[0] + lo[0]; hi[1] = cdims[1] + lo[1];

				hi[0] = hi[0]-1; hi[1] = hi[1]-1;
#if DEBUG>3
				printf ("%d: GET GA_A: lo[0,1] = %d,%d and hi[0,1] = %d,%d\n",proc,lo[0],lo[1],hi[0],hi[1]);
#endif
				NGA_Get(g_a, lo, hi, a, ld);
       
                               /* Perform A^T and store in atrans */
                                for (i=0; i< hi[0] - lo[0]+1; i++)
                                   for (j=0; j< hi[0] - lo[0]+1; j++)
                                        atrans[j*ld[0]+i] = a[i*ld[0]+j];
                     
                               /* A = A + A^T */
                                for (i=0; i< hi[0] - lo[0]+1; i++)
                                   for (j=0; j< hi[0] - lo[0]+1; j++)
                                        a[i*ld[0]+j] += atrans[i*ld[0]+j];                          
      
                                int i = 0;
				for (int n = 0; n < N; n+=cdims[1])
				{
					/* Jeff: This is not necessary if beta=0.0 in DGEMM call */
                                        /* memset (c, 0, sizeof(double) * cdims[0] * cdims[1]); */

					/* B = k x n */
					lo[0] = k; 
                                        lo[1] = n;
					hi[0] = cdims[0] + lo[0] - 1; 
                                        hi[1] = cdims[1] + lo[1] - 1;				
#if DEBUG>3
					printf ("%d: GET_GA_B: lo[0,1] = %d,%d and hi[0,1] = %d,%d\n",proc,lo[0],lo[1],hi[0],hi[1]);
#endif
					NGA_Get(g_b, lo, hi, b, ld);
                                        
                                        if (i>1)
						NGA_NbWait(&(nbh[i%2]));

					//_my_dgemm_ (a, local_N, b, local_N, c, local_N, local_N, local_N, local_N, alpha, beta=1.0);

					/* TODO I am assuming square matrix blocks, further testing/work 
					   required for rectangular matrices */
                                        /* Jeff: I don't know if passing the argument beta=1.0 is valid C.
                                         *       That statement is equivalent to (int)1 aka TRUE is it not? */
					cblas_dgemm ( CblasRowMajor, CblasNoTrans, /* TransA */CblasNoTrans, /* TransB */
							cdims[0] /* M */, cdims[1] /* N */, cdims[0] /* K */, alpha,
							a, cdims[0], /* lda */ b, cdims[1], /* ldb */
							beta, c[i%2], cdims[0] /* ldc */);

					/* C = m x n */
					lo[0] = m; 
                                        lo[1] = n;
					hi[0] = cdims[0] + lo[0] - 1; 
                                        hi[1] = cdims[1] + lo[1] - 1;				
#if DEBUG>3
					printf ("%d: ACC_GA_C: lo[0,1] = %d,%d and hi[0,1] = %d,%d\n",proc,lo[0],lo[1],hi[0],hi[1]);
#endif
					NGA_NbAcc(g_c, lo, hi, c[i%2], ld, &alpha, &nbh[i%2]);
                                        i++;
					//count_acc += 1;
				} /* END LOOP N */
				next_gac = NGA_Read_inc(g_cnt,rdcnt,(long)1);
			} /* ENDIF if count == next */
			count_gac++;
		} /* END LOOP K */
	} /* END LOOP M */

	GA_Sync();
	t2 = GA_Wtime();
	seconds = t2 - t1;
	if (proc == 0)
		printf("Time taken for MM (secs):%lf \n", seconds);

        //printf("Number of ACC: %d\n", count_acc);

#if VERIFY>1
	/* Correctness test - modify data again before this function */
	for (int i = 0; i < NDIMS; i++) {
		lo[i] = 0;
		hi[i] = dims[i]-1;
		ld[i] = dims[i];
	}

	verify(g_a, g_b, g_c, lo, hi, ld, N);
#endif

	/* Clear local buffers */
	ARMCI_Free_local(a);
	ARMCI_Free_local(atrans);
	ARMCI_Free_local(b);
	ARMCI_Free_local(c[0]);
	ARMCI_Free_local(c[1]);

	GA_Sync();

	/* Deallocate arrays */
	GA_Destroy(g_a);
	GA_Destroy(g_b);
	GA_Destroy(g_c);
	GA_Destroy(g_cnt);
	GA_Destroy(g_cnt2);
}

/*
 * Check to see if inversion is correct.
 */
#define TOLERANCE 0.1
void verify(int g_a, int g_b, int g_c, int *lo, int *hi, int *ld, int N) 
{

	double rchk, alpha=1.0, beta=0.0, temp_beta=1.0;
	int g_chk, g_atrans, me=GA_Nodeid();

	g_chk = GA_Duplicate(g_a, "array Check");
	g_atrans = GA_Duplicate(g_a, "array Check");
	if(!g_chk || !g_atrans) GA_Error("duplicate failed",NDIMS);
	GA_Sync();

        GA_Transpose(g_a,g_atrans);
        GA_Add(&alpha, g_a, &temp_beta, g_atrans, g_a);
	GA_Dgemm('n', 'n', N, N, N, 1.0, g_a,
			g_b, 0.0, g_chk);

	GA_Sync();

	alpha=1.0, beta=-1.0;
	GA_Add(&alpha, g_c, &beta, g_chk, g_chk);
	rchk = GA_Ddot(g_chk, g_chk);

	if (me==0) {
		printf("Normed difference in matrices: %12.4e\n", rchk);
		if(rchk < -TOLERANCE || rchk > TOLERANCE)
			GA_Error("Matrix multiply verify failed",0);
		else
			printf("Matrix Mutiply OK\n");
	}

	GA_Destroy(g_chk);
	GA_Destroy(g_atrans);
}


int main(int argc, char **argv) 
{
	int proc, nprocs;
	int M, N, K; /* */
	int blockX_len, blockY_len;
	int heap=3000000, stack=3000000;

	if (argc == 6) {
		M = atoi(argv[1]);
		N = atoi(argv[2]);
		K = atoi(argv[3]);
		blockX_len = atoi(argv[4]);
		blockY_len = atoi(argv[5]);
	}
	else {
		printf("Please enter ./a.out <M> <N> <K> <BLOCK-X-LEN> <BLOCK-Y-LEN>");
		exit(-1);
	}

	MPI_Init(&argc, &argv);

	GA_Initialize();

	if(! MA_init(C_DBL, stack, heap) ) GA_Error("MA_init failed",stack+heap);

	proc   = GA_Nodeid();
	nprocs = GA_Nnodes();

	if(proc == 0) {
		printf("Using %d processes\n", nprocs); 
		printf("\nSize of M, N, K: %d & size of BLOCK: %d \n", M, blockX_len);
                printf("\n**********************************************************\n");
		fflush(stdout);
	}

	matrix_multiply(M, N, K, blockX_len, blockY_len);

	if(proc == 0)
		printf("\nTerminating ..\n");

	GA_Terminate();

	MPI_Finalize();    

	return 0;
}

#include "fakega.h"

int _ga_create( MPI_Comm comm, int dim1, int dim2,  
		MPI_Datatype dtype, GA * ga ) 
{ 
	GA       new_ga; 
	int      chunk2, sizeoftype; 
	MPI_Aint local_size; 
	void     * ga_win_ptr; 

	/* Get a new structure */ 
	new_ga = (GA)malloc( sizeof(struct _GA) );   
	if (!new_ga) return 0; 

	/* Determine size of GA memory */ 
	chunk2 = ((dim2 / nprocs) == 0 ? 1 : (dim2 / nprocs));
	
	/* Require size to exactly divide dim2 */ 
	if (((dim2 % nprocs) != 0) && (dim2 != 1)) 
		MPI_Abort( comm, 666 ); 
	
	MPI_Type_size( dtype, &sizeoftype ); 
	local_size = dim1 * chunk2 * sizeoftype; 

	/* Allocate memory my ga_win and create window */ 
	MPI_Win_allocate (local_size, sizeoftype, MPI_INFO_NULL, comm, &ga_win_ptr, &new_ga->ga_win);
	memset(ga_win_ptr, 0, local_size);	
	MPI_Win_lock_all( MPI_MODE_NOCHECK, new_ga->ga_win ); 

	MPI_Barrier (MPI_COMM_WORLD);
	
	/* Save other data and return */ 
	new_ga->dtype      = dtype; 
	new_ga->dtype_size = sizeoftype; 
	new_ga->dim1       = dim1; 
	new_ga->dim2       = dim2; 
	new_ga->chunk2     = chunk2; 
	new_ga->comm	   = comm;
	new_ga->win_base   = ga_win_ptr;
	*ga                = new_ga; 
	
	return 1; 
}

double _ga_wtime (void)
{
	return MPI_Wtime();

}

int _ga_destroy( GA ga ) 
{
	MPI_Win_unlock_all(ga->ga_win);
	MPI_Win_free( &ga->ga_win ); 
	free(ga); 

	return 0; 
}

int _ga_zero ( GA ga )
{
	size_t size = ga->dim1 * ga->dtype_size * ga->chunk2;
	memset (&ga->win_base, 0, size);

	return 1;
}
/* Notes: Sayan: Modify this */
int _ga_duplicate( GA ga, GA * dup )
{
	int status = _ga_create(ga->comm, ga->dim1, ga->dim2, ga->dtype, dup ); 
	return status;
}	

int _ga_transfer (GA ga, int ilo, int jlo, int ihigh, int jhigh, void * buf, int * ld, transfer_type xtype, MPI_Request * hdl) 
{
	int jcur, jlast, rank, count; 
	MPI_Aint disp;
	//MPI_Request req;
       	//int jfirst;	
	
	jcur = jlo; 
	while (jcur <= jhigh) { 
		rank   = jcur / ga->chunk2; 
		//jfirst = rank * ga->chunk2; 
		jlast  = (rank + 1) * ga->chunk2 - 1; 
		if (jlast > jhigh) jlast = jhigh; 
		count = jlast - jcur + 1;
#if DEBUG>2
		printf ("%d: jcur = %d, jhigh = %d and jlast = %d\n",rank,jcur,jhigh,jlast);
		printf ("%d: count = %d\n",rank,count);
#endif 
		for (int i = ilo; i <= ihigh; i++) {
			disp = i * count;	
#if DEBUG>3
			printf ("%d: disp = %ld\n",rank,disp);
#endif 		
			switch (xtype) {
				case PUT:	
					MPI_Put( buf, count, ga->dtype, rank, disp, 
							count, ga->dtype, ga->ga_win ); 
					MPI_Win_flush_local (rank, ga->ga_win);
					break;
					/*
					   MPI_Rput( &buf, count, ga->dtype, rank, disp, 
					   count, ga->dtype, ga->ga_win, &req ); 
					   MPI_Wait (&req, MPI_STATUS_IGNORE);
					   break;
					 */
				case GET:
					MPI_Get( buf, count, ga->dtype, rank, disp, 
							count, ga->dtype, ga->ga_win ); 
					MPI_Win_flush_local (rank, ga->ga_win);
					break;
					/*	
						MPI_Rget( &buf, count, ga->dtype, rank, disp, 
						count, ga->dtype, ga->ga_win, &req ); 
						MPI_Wait (&req, MPI_STATUS_IGNORE);
						break;
					 */
				case ACC:
					MPI_Accumulate( buf, count, ga->dtype, rank, disp, 
							count, ga->dtype, MPI_SUM, ga->ga_win ); 
					MPI_Win_flush_local (rank, ga->ga_win);
					break;
					/*
					   MPI_Raccumulate( &buf, count, ga->dtype, rank, disp, 
					   count, ga->dtype, MPI_SUM, ga->ga_win, &req ); 
					   MPI_Wait (&req, MPI_STATUS_IGNORE);
					   break;
					 */
				case NbACC:
					MPI_Raccumulate( buf, count, ga->dtype, rank, disp, 
							count, ga->dtype, MPI_SUM, ga->ga_win, hdl ); 
					break;
					/*
					   MPI_Raccumulate( &buf, count, ga->dtype, rank, disp, 
					   count, ga->dtype, MPI_SUM, ga->ga_win, &req ); 
					   MPI_Wait (&req, MPI_STATUS_IGNORE);
					   break;
					 */
			}
			/* resize */
			buf = (void *)( ((char *)buf) + count *  ga->dtype_size ); 
		}
		jcur = jlast + 1; 
	}

       	return 0;	
}

int _nga_put( GA ga, int * lo, int * hi, void * buf, int * ld ) 
{ 
	return _ga_transfer (ga, lo[0], lo[1], hi[0], hi[1], buf, ld, PUT, NULL);
} 

int _nga_get( GA ga, int * lo, int * hi, void * buf, int * ld ) 
{ 
	return _ga_transfer (ga, lo[0], lo[1], hi[0], hi[1], buf, ld, GET, NULL);
} 

int _nga_acc( GA ga, int * lo, int * hi, void * buf, int * ld ) 
{ 
	return _ga_transfer (ga, lo[0], lo[1], hi[0], hi[1], buf, ld, ACC, NULL);
} 

int _nga_nbacc( GA ga, int * lo, int * hi, void * buf, int * ld, MPI_Request * hdl ) 
{ 
	return _ga_transfer (ga, lo[0], lo[1], hi[0], hi[1], buf, ld, NbACC, hdl);
} 

long long _ga_read_inc( GA ga, int * subscript, long inc ) 
{
        /* TODO offset=0 only works if subscript is an array of zeros */
        /* since GAs are hard-coded to have only 2 dimensions, we don't 
         * need to loop over 0:(ga->ndim-1) */
        //assert(subscript[0]==0 && subscript[1]==0);
        MPI_Aint offset = (MPI_Aint)subscript[0] * (ga->chunk2) + subscript[1];

        long long otemp = 0; /* will contain value of remote counter _prior_ to increment */
	MPI_Fetch_and_op (&inc, &otemp, MPI_LONG, 0, offset, MPI_SUM, ga->ga_win);
        /* no need to flush+unlock - just remove lock+unlock once lock_all used throughout */
	MPI_Win_flush_local (0, ga->ga_win);
	
	return otemp;
}

void _ga_error (char * str, int code)
{
	printf ("%s\n",str);
	MPI_Abort (MPI_COMM_WORLD, code);
}

void _ga_flush (GA ga) 
{
	MPI_Win_flush_all (ga->ga_win);
	return;
}

void _ga_sync (void)
{
	MPI_Barrier (MPI_COMM_WORLD);
}
/* c = alpha * a  +  beta * b */
void _ga_add(double alpha, GA g_a, double beta, GA g_b, GA g_c)
{
	for (int i = 0; i < g_a->dim1; i++)
		for (int j = 0; j < g_a->chunk2; j++)
			((double *)g_c->win_base)[i*g_a->dim1 + j] = 
				((double *)g_a->win_base)[i*g_a->dim1 + j] * alpha + 
				((double *)g_b->win_base)[i*g_a->dim1 + j] * beta;
	
	return;
}
/*
   MPI_Alltoall(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
   void* recvbuf, int recvcount, MPI_Datatype recvtype,
   MPI_Comm comm)
 */
/* B = A^T */
void _ga_transpose( GA g_a, GA g_b)
{
	if (g_a->win_base != g_b->win_base)
		MPI_Alltoall(g_a->win_base, (g_a->dim1 * g_a->chunk2), g_a->dtype,
				g_b->win_base, (g_b->dim1 * g_b->chunk2), g_b->dtype, g_a->comm);
	else
		MPI_Alltoall(MPI_IN_PLACE, 0, MPI_CHAR,
				g_b->win_base, (g_b->dim1 * g_b->chunk2), g_b->dtype, g_a->comm);
	return;
}

void _ga_dgemm(char transa[1], char transb[1], int m, int n, int k, double alpha, GA g_a, GA g_b, double beta, GA g_c )
{
	void * row_b;
	MPI_Alloc_mem (g_a->dtype_size * g_a->dim1, MPI_INFO_NULL, &row_b);
	/* 
	   ensure square matrices and dimensions of all global 
	   arrays are equivalent (equal chunk distributions)
	*/  
       	/* expecting dim1 == dim2 && 
	   (dim1 | dim2) % chunk2 (of g_a == g_b == g_c) == 0 */
	if ((g_a->dim1 % g_a->chunk2) != 0)	
		_ga_error ("At present, we expect dim1 == dim2 and (dim1 | dim2) MOD chunk2 == 0", g_a->chunk2);

	for (int i = 0; i < g_a->dim1; i++) {
		/* gather row chunks 
		 int MPI_Allgather(const void* sendbuf, int sendcount,
		 MPI_Datatype sendtype, void* recvbuf, int recvcount,
		 MPI_Datatype recvtype, MPI_Comm comm)
		 */
		MPI_Allgather (((char *)g_b->win_base + i*g_a->dtype_size), g_b->chunk2, g_b->dtype, &row_b, 
				g_b->chunk2, g_b->dtype, g_b->comm);

		for (int j = 0; j < g_a->chunk2; j++) {
			((double *)g_c->win_base)[i*g_a->chunk2 + j] = ((double *)g_c->win_base)[i*g_a->chunk2 + j] * beta + 
				alpha * ((double *)row_b)[j] * ((double *)g_a->win_base)[i*g_a->chunk2 + j];				
		}
	}
	MPI_Free_mem (row_b);
	return;
}

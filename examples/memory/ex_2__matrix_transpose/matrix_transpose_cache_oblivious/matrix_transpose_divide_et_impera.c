/* ────────────────────────────────────────────────────────────────────────── *
 │                                                                            │
 │ This file is part of the exercises for the Lectures on                     │
 │   "Foundations of High Performance Computing"                              │
 │ given at                                                                   │
 │   Master in HPC and                                                        │
 │   Master in Data Science and Scientific Computing                          │
 │ @ SISSA, ICTP and University of Trieste                                    │
 │                                                                            │
 │ contact: luca.tornatore@inaf.it                                            │
 │                                                                            │
 │     This is free software; you can redistribute it and/or modify           │
 │     it under the terms of the GNU General Public License as published by   │
 │     the Free Software Foundation; either version 3 of the License, or      │
 │     (at your option) any later version.                                    │
 │     This code is distributed in the hope that it will be useful,           │
 │     but WITHOUT ANY WARRANTY; without even the implied warranty of         │
 │     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          │
 │     GNU General Public License for more details.                           │
 │                                                                            │
 │     You should have received a copy of the GNU General Public License      │
 │     along with this program.  If not, see <http://www.gnu.org/licenses/>   │
 │                                                                            │
 * ────────────────────────────────────────────────────────────────────────── */


#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 201112L)    // c11
#    define _XOPEN_SOURCE 700
#  elif (__STDC_VERSION__ >= 199901L)  // c99
#    define _XOPEN_SOURCE 600
#  else
#    define _XOPEN_SOURCE 500          // c90
#  endif
#endif


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#if _XOPEN_SOURCE >= 600
#  include <strings.h>
#endif
#include <sys/resource.h>
#include <sys/times.h>
#include <time.h>
#include <math.h>

#include "../mypapi.h"

#define CACHE_OBLIVIOUS 0
#define NREPETITIONS 5
#define CONTIGUOUS 0

#if !defined(WRITE)
#define WRITE CONTIGUOUS
#endif


#define CPU_TIME ({struct  timespec ts; clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ),	\
                                          (double)ts.tv_sec +           \
                                          (double)ts.tv_nsec * 1e-9;})


// We may recurse down to 1x1; however, in practice, function call overhead would
// outweighs cache benefits at small sizes.
// We stop at a block size that comfortably fits in the L1 cache.


#if !defined(DTYPE)
// chancge the data type at compile time by
//# -DDTYPE=double
//# -DDTYPE=float
//# -DDTYPE=uint32_t
// and so on

// default to uint32_t
#define DTYPE uint32_t
#endif

typedef DTYPE dtype;


// Base case: Standard naive transpose for a small block
void transpose_block( const dtype *src, dtype * restrict dst, 
                      int r_start, int r_end, 
                      int c_start, int c_end, 
                      int n, int m )
{
  // src is n x m, dst is m x n

 #if (WRITE == CONTIGUOUS)
 #warning "compiled with contiguous write"

  for ( int j = c_start; j < c_end; j++ )
    for ( int i = r_start; i < r_end; i++ )
      dst [ j * n + i ] = src [ i * m + j ];

 #else // WRITE == STRIDED
 #warning "compiled with strided write"

  for ( int i = r_start; i < r_end; i++ )
    for ( int j = c_start; j < c_end; j++ )
      dst[j * n + i] = src[i * m + j];

 #endif
  
  return;
}

// Recursive function to traverse memory in a Z-order/divide-and-conquer pattern
void transpose_cache_oblivious( const dtype *src, dtype * restrict dst, 
                                int r_start, int r_end, 
				int c_start, int c_end, 
				int n, int m, int threshold )
{
  int r_len = r_end - r_start;
  int c_len = c_end - c_start;

  // Base case: If the block is small enough, process it directly
  if (r_len <= threshold && c_len <= threshold )
    {
      transpose_block(src, dst, r_start, r_end, c_start, c_end, n, m);
    } 
  // Divide and conquer: split the longest dimension
  else if (r_len >= c_len)
    {
      int r_mid = r_start + (r_len >> 1);
      transpose_cache_oblivious ( src, dst, r_start, r_mid, c_start, c_end, n, m, threshold );
      transpose_cache_oblivious ( src, dst, r_mid, r_end, c_start, c_end, n, m, threshold );
    } 
  else
    {
      int c_mid = c_start + (c_len >> 1);
      transpose_cache_oblivious ( src, dst, r_start, r_end, c_start, c_mid, n, m, threshold );
      transpose_cache_oblivious ( src, dst, r_start, r_end, c_mid, c_end, n, m, threshold );
    }
  
  return;
}



int main ( int argc, char **argv )
{
  if ( (argc > 1) && 
       ((strcmp(*(argv+1), "-h") == 0) ||
	(strcmp(*(argv+1), "--help") == 0)) ) {
    printf("arguments: MODE ( 0 = cache oblivious, !=0 standard) N_ROWS N_COLS THRESHOLD (for std block)\n");
    return 0; }

  PAPI_INIT;
  
  int mode      = (argc > 1? atoi(*(argv+1)) : CACHE_OBLIVIOUS); // running mode
  int n         = (argc > 2? atoi(*(argv+2)) : 4000); // rows
  int m         = (argc > 3? atoi(*(argv+3)) : 3000); // columns
  int threshold = (argc > 4? atoi(*(argv+4)) : 64); // columns

  // Allocate 1D arrays to represent 2D contiguous matrices
  dtype *src = (dtype *)malloc(n * m * sizeof(dtype));
  dtype *dst = (dtype *)malloc(m * n * sizeof(dtype));
  
  if (!src || !dst) {
    fprintf(stderr, "Memory allocation failed.\n");
    return 1;
  }
  
  // Initialize source matrix
  unsigned int size = n*m;
  for ( unsigned int i = 0; i < size; i++ )
    src[i] = (dtype)i;

  double timing_max;
  double timing_sum = 0;
  
  if ( mode == CACHE_OBLIVIOUS )
    {
      // warm up
      timing_max = CPU_TIME;
      transpose_cache_oblivious ( src, dst, 0, n, 0, m, n, m, threshold );
      timing_max = CPU_TIME - timing_max;

      // measure
      for ( int r = 0; r < NREPETITIONS; r++ )
	{
	  double this_time = CPU_TIME;
	  PAPI_START_CNTR;
	  transpose_cache_oblivious ( src, dst, 0, n, 0, m, n, m, threshold );
	  PAPI_STOP_CNTR;
	  this_time = CPU_TIME - this_time;
	  
	  timing_max  = ( timing_max < this_time  ? this_time : timing_max ); 
	  timing_sum += this_time;
	}
    }
  else
    {
      // warm up
      timing_max = CPU_TIME;
      transpose_block ( src, dst, 0, n, 0, m, n, m );
      timing_max = CPU_TIME - timing_max;

      // measure
      for ( int r = 0; r < NREPETITIONS; r++ )
	{
	  double this_time = CPU_TIME;
	  PAPI_START_CNTR;
	  transpose_block ( src, dst, 0, n, 0, m, n, m );
	  PAPI_STOP_CNTR;
	  this_time = CPU_TIME - this_time;
	  
	  timing_max  = ( timing_max < this_time  ? this_time : timing_max ); 
	  timing_sum += this_time;
	}
    }
  
  printf ( "elapsed time (avg out of %d measures): %g s\n",
	   NREPETITIONS - 1,
	   (timing_sum - timing_max) / (NREPETITIONS-1) );


 #if defined(USE_PAPI)
  {
    int event_codes[PAPI_EVENTS_NUM];
    int nevents;
    PAPI_list_events( papi_EventSet, event_codes, &nevents);
    for( int jj = 0; jj < PAPI_EVENTS_NUM; jj++) {
      char name[PAPI_MAX_STR_LEN+1];
      PAPI_event_code_to_name(event_codes[jj], name);
      printf("event %*s : %10llu\n", PAPI_MAX_STR_LEN, name, papi_values[jj]/(NREPETITIONS)); }
  }
 #endif

  
  // Optional: Verify a few elements
  int fail = (src[2 * m + 3] != dst[3 * n + 2]);
  if ( fail )
    printf("some problem occured!\n");
  
  
  free(src);
  free(dst);
  
  return 0;
}

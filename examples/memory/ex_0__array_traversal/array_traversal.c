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


#define _XOPEN_SOURCE 700  // c11

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

#define CPU_TIME ({struct  timespec ts; clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ),	\
                                          (double)ts.tv_sec +           \
                                          (double)ts.tv_nsec * 1e-9;})


int main(int argc, char** argv)
{
  const size_t N = ( argc > 1 ? strtoull( argv[1], NULL, 10) : 8192 ); // ~512MB for N=8192 (double)

  const int REPEAT = 3;

  //
  // Row-major allocation: traversing [i][j], is is the j fastest-changing index and 
  // indicates the memory-contiguous dimension
  // 
  
  double *A = aligned_alloc ( 64, N * N * sizeof(double) );
  if ( A == NULL )
    {
      char buffer[1000];
      sprintf( &buffer[0], "error in allocating %lu Mbytes\n",
	       N*N*sizeof(double) / (1024*1024) );
      perror ( buffer );
      return 1;
    }

  for( size_t i = 0; i < N*N; i++ )
    A[i] = (double)(i&255);
  
  // Warm-up the cache
  volatile double sink = 0;
  for ( size_t i = 0; i < N*N; i++ )
    sink += A[i];
  
  double best_row = 1e9, best_col = 1e9;
  double srow = 0, scol = 0;
  
  for ( int  r = 0; r < REPEAT; r++ )
    // row-major: cache-friendly
    {
      double tstart = CPU_TIME;
      double sum = 0;
      for ( size_t i = 0; i < N; i++ )
	{
	  size_t base = i*N;
	  for ( size_t j = 0; j < N; j++)
	    sum += A[base + j];            
	}
      double timing = CPU_TIME - tstart;
      best_row = (best_row < timing ? best_row : timing );
      srow = sum;
    }
  
  for (int r = 0; r < REPEAT; r++ )
    // column-major (strided) : cache unfriendly
    {
      double tstart = CPU_TIME;
      double sum=0;
      for ( size_t j = 0; j < N; j++ )
	{               
	  for ( size_t i = 0; i < N; i++ )
	    sum += A[i*N + j];
	}
      double timing = CPU_TIME - tstart;
      best_col = ( best_col < timing ? best_col : timing );
      scol = sum;
    }
  
  double bytes = (double)(N*N) * sizeof(double);
  
  printf ("N = %zu\n"
	  "Row-major:    %.3f s, %.2f GB/s\n"
	  "Column-major: %.3f s, %.2f GB/s\n",
	  N,
	  best_row, bytes/best_row/1e9,
	  best_col, bytes/best_col/1e9);
  
  // Prevent “optimization”
  fprintf ( stderr, "just to know: %.1f %.1f\n", srow, scol);
  
  // release memory
  free(A);
  
  return 0;
}

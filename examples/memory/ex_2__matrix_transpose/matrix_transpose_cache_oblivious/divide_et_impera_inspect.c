
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


void transpose_cache_oblivious( int r_start, int r_end, 
				int c_start, int c_end, 
				int threshold )
{
  int r_len = r_end - r_start;
  int c_len = c_end - c_start;

  // Base case: If the block is small enough, process it directly
  if (r_len <= threshold && c_len <= threshold )
    {
      printf("%4d -> %4d   %4d -> %4d    final\n",
      r_start, r_end, c_start, c_end );
    } 
  // Divide and conquer: split the longest dimension
  else if (r_len >= c_len)
    {
      int r_mid = r_start + (r_len >> 1);
      
      printf("\t  %4d -> %4d   %4d -> %4d\n",
	     r_start, r_mid, c_start, c_end );

      printf("\t  %4d -> %4d   %4d -> %4d\n",
	     r_mid, r_end, c_start, c_end );
      
      transpose_cache_oblivious ( r_start, r_mid, c_start, c_end, threshold );
      transpose_cache_oblivious ( r_mid, r_end, c_start, c_end, threshold );
    } 
  else
    {
      int c_mid = c_start + (c_len >> 1);

      printf("\t  %4d -> %4d   %4d -> %4d\n",
	     r_start, r_end, c_start, c_mid );
      
      printf("\t  %4d -> %4d   %4d -> %4d\n",
	     r_start, r_end, c_mid, c_end );

      transpose_cache_oblivious ( r_start, r_end, c_start, c_mid, threshold );
      transpose_cache_oblivious ( r_start, r_end, c_mid, c_end, threshold );
    }
  
  return;
}


int main ( int argc, char **argv )
{
  int rows = ( argc > 1 ? atoi(*(argv+1)) : 1024 );
  int cols = ( argc > 2 ? atoi(*(argv+2)) : 1024 );
  int thrs = ( argc > 3 ? atoi(*(argv+3)) : 32 );

  transpose_cache_oblivious ( 0, rows, 0, cols, thrs );
  
  return 0;
}

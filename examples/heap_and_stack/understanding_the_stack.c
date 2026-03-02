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

// A global pointer used to point to a dead stack frame
int *dangling_ptr = NULL;



void function_C(int overwrite_val, int **trap)
{
  register long long int myRSP __asm__("rsp");
  register long long int myRBP __asm__("rbp");
  
  // This variable will likely occupy the exact same memory address 
  // that function_B's local variable used previously.
  int local_c = overwrite_val; 
  
  printf("\n--- Inside function_C ---\n");
  printf("[funcC] RSP points to: %p\n"
	 "[funcC] RBP points to: %p\n", (void*)myRSP, (void*)myRBP);
  printf("[funcC] Address of local_c: %p (value: %d)\n", (void*)&local_c, local_c);
  printf("[funcC] We just overwrote the old stack frame!\n");
  
  *trap = &local_c;
}

void function_B(int *caller_var_ptr)
{
  register long long int myRSP __asm__("rsp");
  register long long int myRBP __asm__("rbp");
  
  int local_b = 42;
  
  printf("\n--- Inside function_B ---\n");
  printf("[funcB] RBP points to: %p\n"
	 "[funcB] RSP points to: %p\n",
	 (void*)myRSP, (void*)myRBP);
  
  // Concept 1: The stack grows downwards.
  // - rbp and rsp have smalle values than in main
  // - local_b has a lower memory address than main's local_main,
  //   in between of rbp and rsp.
  printf("[funcB] Address of local_b: %p\n", (void*)&local_b);
  if ((void*)&local_b < (void*)caller_var_ptr) {
    printf("[funcB] -> PROOF: Stack grew DOWNWARDS (local_b addr < local_main addr).\n");
  }
  
  // Concept 2: Accessing the caller's stack.
  // We dereference the pointer passed from main to modify main's local variable.
  printf("\n[funcB] Modifying caller's stack...\n");
  *caller_var_ptr = 999;
  printf("[funcB] Caller's variable modified successfully.\n");
  
  // Concept 3 Setup: Saving a pointer to our local stack frame before returning.
  // WARNING: This is extremely dangerous in real code!
  dangling_ptr = &local_b;
  printf("[funcB] Saved address of local_b to global dangling_ptr.\n");
}

int main( void )
{
  register long long int myRSP __asm__("rsp");
  register long long int myRBP __asm__("rbp");
  
  int local_main = 10;
  
  printf("--- Inside main ---\n");
  printf("[main] RSP points to: %p\n"
	 "[main] RBP points to: %p\n", (void*)myRSP, (void*)myRBP );
  printf("[main] Address of local_main: %p (initial value: %d)\n", (void*)&local_main, local_main);
  
  // Pass the address of main's local variable down the stack to function_B
  function_B(&local_main);
  
  printf("\n--- Back in main ---\n");
  // Proof of Concept 2
  printf("[main] Value of local_main is now: %d (modified by function_B!)\n", local_main);
  
  // Proof of Concept 3 (Part 1)
  printf("\n[main] Let's look at the dangling pointer left behind by function_B...\n");
  printf("[main] dangling_ptr points to %p\n", (void*)dangling_ptr);
  printf("[main] Value at dangling_ptr right now: %d (It might still be 42!)\n", *dangling_ptr);
  
  // Call another function to overwrite the memory where function_B used to be
  int *ptr_to_trap;
  function_C(8888, &ptr_to_trap);
  
  printf("\n--- Back in main again ---\n");
  // Proof of Concept 3 (Part 2)
  printf("[main] Let's check the dangling pointer again after function_C ran.\n");
  printf("[main] Value at dangling_ptr is now: %d\n", *dangling_ptr);
  printf("[main] -> PROOF: function_B's stack was destroyed and overwritten by function_C!\n");

  printf("[main] and now.. let's fall in the trap!\n");
  printf("[main] the value of trap is %d\n", *ptr_to_trap );
	 
  
  return 0;
}

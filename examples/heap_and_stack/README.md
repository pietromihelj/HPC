# Examples on how the stak works

#### `explore_how_bytes_are_stored.c`

This code is meant as an exercise to make you practicing with single bytes managing.
The excuse is to verify what is the relative placement of least- and most- significant bytes in memory.

#### `understanding_the_stack.c`

This code is about verifying that the stack grows downwards and that accessing the stack of a funtion after the function’s return is undefined behaviour

#### `stack_overflow.c`

This code performs the summation of an array’s elements by recursion. That is done purposely because when the size of the array grows the recursion deepens and beyond some limit the stack is exhausted and the code crashes (500000 elements should be sufficient on most systems).
When compiling the code with `-O3` it does not crash because the compiler transforms the recursion into an iteration. The code `stack_overflow.inspect_recursion.c` allows to verify that because with `-O3` the content of the `BP` register does not change, which means that we are still in the same stack frame.

#### `stack_smash.c`

This code, via a loop over an array placed on the stack, smash the base of the stack because the iteration space of the loop is larger than the array’s size.
Typically the compiler injects guardian bytes that allows the issuing of a specific warning about the stack smashing.
Compiling with `-fno-stack-protector` (for `gcc`) disable that, and you simply get a crash.

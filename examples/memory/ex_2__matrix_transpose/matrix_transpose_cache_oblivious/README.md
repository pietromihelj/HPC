# Examples of cache-oblivious algorithm

In this folder we examine two cache-oblivious algorithm and draw some interesting conclusions - see the file `lecture_matrix_transpose_cache_oblivious.md`.

As first, let’s remind the difference between cache-aware and cache-oblivious algorithms. The core difference lies in whether the algorithm explicitly relies on the hardware's specific cache parameters (like total cache capacity and cache line size) either directly or by tuning internal parameters.

- **Cache-Aware Algorithms:** These algorithms are explicitly tuned to the specific hardware they run on. They use known hardware parameters to optimize memory access by breaking data into chunks that perfectly fit the target cache. A classic example is loop blocking/tiling in matrix operations, where the block size is hardcoded to match a specific CPU's L1 cache. If the hardware changes, the code must be re-tuned to maintain optimal performance.
- **Cache-Oblivious Algorithms:** These algorithms are designed to optimally use the CPU cache *without* knowing its specific size or structure. They contain no hardware-specific tuning variables. Instead, they rely on recursive, divide-and-conquer strategies—like the $Z$-order (Morton) space-filling curve you are using in your code. By continually dividing the problem into smaller and smaller sub-problems, they naturally hit a point where the data fits perfectly into any level of the memory hierarchy (L1, L2, L3) on any machine, optimizing cache usage automatically.


​	

| File                                  | Description                                                  |
| ------------------------------------- | ------------------------------------------------------------ |
| `matrix_transpose_divide_et_impera.c` | transposition by recursively halving the longest dimension - reproduce a tiling pattern |
| `matrix_transpose_z-order.v1.c`       | transposition by traversing via $z$-order<br />- with cache resonance and memory aliasing |
| `matrix_transpose_z-order.v2.c`       | transposition by traversing via $z$-order<br />- memory padding, no memory aliasing |
| `matrix_transpose_z-order.v1.fixed.c` | transposition by traversing via $z$-order<br />- with cache resonance and fixed memory aliasing |
| `morton_inspect.c`                    | small code to inspect the $z$-order traversal                |
| `divide_et_impera_inspect.c`          | small code to inspect the recursive order traversal          |


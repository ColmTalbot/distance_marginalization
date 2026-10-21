# distmarg Python benchmark report

## Array backends (values and gradients)

Prior usf_100_5000; posterior-like inputs (quadrature regime); µs per call and ns per element.

| backend | phase | batch | value µs/call | value ns/elem | grad µs/call | grad ns/elem | first call (compile) |
|---|---|---|---|---|---|---|---|
| numpy → Rust (scalar) | False | 1 | 1.9 | 1918 | – | – | – |
| numpy → Rust (array) | False | 1 | 5.1 | 5134 | – | – | – |
| numpy, pure Python | False | 1 | 550.8 | 550808 | – | – | – |
| JAX jit (CPU) | False | 1 | 26.0 | 25979 | 40.3 | 40287 | 0.59 s |
| torch eager (CPU) | False | 1 | 2057.2 | 2057229 | 5429.5 | 5429480 | – |
| numpy → Rust (array) | False | 100 | 156.2 | 1562 | – | – | – |
| numpy, pure Python | False | 100 | 854.9 | 8549 | – | – | – |
| JAX jit (CPU) | False | 100 | 187.6 | 1876 | 600.6 | 6006 | 0.55 s |
| torch eager (CPU) | False | 100 | 2353.3 | 23533 | 6410.0 | 64100 | – |
| numpy → Rust (array) | False | 10000 | 2234.1 | 223 | – | – | – |
| numpy, pure Python | False | 10000 | 34038.9 | 3404 | – | – | – |
| JAX jit (CPU) | False | 10000 | 3922.9 | 392 | 8449.3 | 845 | 0.58 s |
| torch eager (CPU) | False | 10000 | 37845.1 | 3785 | 77146.0 | 7715 | – |
| numpy → Rust (scalar) | True | 1 | 4.2 | 4177 | – | – | – |
| numpy → Rust (array) | True | 1 | 7.0 | 7035 | – | – | – |
| numpy, pure Python | True | 1 | 668.4 | 668435 | – | – | – |
| JAX jit (CPU) | True | 1 | 36.0 | 36043 | 53.1 | 53106 | 0.86 s |
| torch eager (CPU) | True | 1 | 2642.8 | 2642792 | 7783.9 | 7783920 | – |
| numpy → Rust (array) | True | 100 | 408.1 | 4081 | – | – | – |
| numpy, pure Python | True | 100 | 1236.6 | 12366 | – | – | – |
| JAX jit (CPU) | True | 100 | 371.3 | 3713 | 931.7 | 9317 | 2.38 s |
| torch eager (CPU) | True | 100 | 3381.6 | 33816 | 9134.0 | 91340 | – |
| numpy → Rust (array) | True | 10000 | 5767.1 | 577 | – | – | – |
| numpy, pure Python | True | 10000 | 60621.8 | 6062 | – | – | – |
| JAX jit (CPU) | True | 10000 | 6322.7 | 632 | 12791.5 | 1279 | 1.19 s |
| torch eager (CPU) | True | 10000 | 47182.5 | 4718 | 99592.8 | 9959 | – |
| numpy → Rust (array) | False | 1000000 | 91534.2 | 92 | – | – | – |
| numpy, pure Python | False | 1000000 | 3490923.0 | 3491 | – | – | – |
| JAX jit (CPU) | False | 1000000 | 468558.8 | 469 | 852905.3 | 853 | 1.01 s |
| torch eager (CPU) | False | 1000000 | 913755.4 | 914 | 2988791.4 | 2989 | – |
| numpy → Rust (array) | True | 1000000 | 237188.7 | 237 | – | – | – |
| numpy, pure Python | True | 1000000 | 6027230.7 | 6027 | – | – | – |
| JAX jit (CPU) | True | 1000000 | 771224.3 | 771 | 1294898.5 | 1295 | 1.83 s |
| torch eager (CPU) | True | 1000000 | 1370231.8 | 1370 | 3755649.5 | 3756 | – |

Peak resident memory for the 10^6 run (all backends, values and gradients): 18 GB. The JAX/torch where-path evaluates every branch on every element, with 120 quadrature nodes each.

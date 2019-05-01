# Parallel Ford Fulkerson on GPU
This repo contains parallel and serial ford fulkerson implementation. Parallel algorithm is implemented using CUDA with compute
capability 3.5.

### Code structure
1. src - contains serial and parallel implementation
2. dataset - contains graphs of varying size to find maxflow of.

### Code execution
- Compile src/parallel_ford_fulkerson.cu with nvcc
  ```
  nvcc src/parallel_ford_fulkerson.cu
  ```
- Execute binary generated with dataset file and no. of vertices in that dataset
  ```
  a.out dataset/10000v.in 10000
  ```

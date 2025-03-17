# SQPCPU - Sequential Quadratic Programming in C++

This repository contains a C++ implementation of a Sequential Quadratic Programming (SQP) solver for trajectory optimization, with Python bindings.

## New Feature: Parallel Batch Processing

The library now supports parallel batch processing of multiple SQP problems using multithreading. This is implemented in the `BatchThneed` class.

## Building the Library

### Prerequisites

- CMake (>= 3.12)
- C++ compiler with C++17 support
- Pinocchio
- OsqpEigen
- Eigen3
- Python with NumPy (for Python bindings)

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/sqpcpu.git
cd sqpcpu

# Create a build directory
mkdir build
cd build

# Configure and build
cmake ..
make -j4

# Optionally install
make install
```

## Using the Batch SQP Solver

### C++ API

```cpp
#include "batch_thneed.hpp"

// Create a batch solver
sqpcpu::BatchThneed batch_solver(urdf_filename, batch_size, N, dt, max_qp_iters, num_threads);

// Prepare batch inputs
std::vector<Eigen::VectorXd> xs_batch;
std::vector<Eigen::VectorXd> eepos_g_batch;

// Fill batch inputs
// ...

// Run batch SQP
batch_solver.batch_sqp(xs_batch, eepos_g_batch);

// Get results
std::vector<Eigen::VectorXd> results = batch_solver.get_results();
```

### Python API

```python
import pysqpcpu
import numpy as np

# Create a batch solver
batch_solver = pysqpcpu.BatchThneed(
    urdf_filename=urdf_filename,
    batch_size=batch_size,
    N=N,
    dt=dt,
    max_qp_iters=max_qp_iters,
    num_threads=num_threads
)

# Prepare batch inputs
xs_batch = []
eepos_g_batch = []

# Fill batch inputs
# ...

# Run batch SQP
batch_solver.batch_sqp(xs_batch, eepos_g_batch)

# Get results
results = batch_solver.get_results()
```

## Examples

The repository includes examples for both C++ and Python:

- C++: `examples/batch_example.cpp`
- Python: `examples/batch_example.py`

To run the C++ example:

```bash
./build/batch_example
```

To run the Python example:

```bash
python examples/batch_example.py
```

## Performance

The batch processing implementation can provide significant speedup compared to sequential execution, especially for larger batch sizes. The actual speedup depends on:

1. The number of available CPU cores
2. The complexity of each SQP problem
3. The batch size

In our tests, we've observed speedups of up to Nx on an N-core machine for compute-bound problems.

## Implementation Details

The batch processing is implemented using a thread pool that manages a fixed number of worker threads. Each SQP problem is submitted as a task to the thread pool, and the results are collected once all tasks are complete.

The implementation ensures that:

1. Each Thneed solver instance runs in its own thread
2. The number of threads is configurable (defaults to hardware concurrency)
3. Resources are properly managed and released

## License

[Your License Here] 
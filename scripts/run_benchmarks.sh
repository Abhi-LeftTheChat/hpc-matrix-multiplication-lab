#!/bin/bash
# PGC Benchmark Automation Script for C and C++ Implementations

set -e

echo "=========================================================="
echo " Parallel & Grid Computing (PGC) Matrix Multiplication "
echo "=========================================================="

# 1. Compile C & C++ Sequential & OpenMP programs
echo "[+] Compiling C Sequential implementation..."
gcc -O2 src/matrix_sequential.c -o src/matrix_sequential_c

echo "[+] Compiling C++ Sequential implementation..."
g++ -O2 src/matrix_sequential.cpp -o src/matrix_sequential_cpp

echo "[+] Compiling C OpenMP implementation..."
gcc -O2 -fopenmp src/matrix_openmp.c -o src/matrix_openmp_c

echo "[+] Compiling C++ OpenMP implementation..."
g++ -O2 -fopenmp src/matrix_openmp.cpp -o src/matrix_openmp_cpp

# 2. Compile MPI (C & C++) if available
if command -v mpicc &> /dev/null; then
    echo "[+] Compiling C MPI implementation..."
    mpicc -O2 src/matrix_mpi.c -o src/matrix_mpi_c
fi

if command -v mpicxx &> /dev/null; then
    echo "[+] Compiling C++ MPI implementation..."
    mpicxx -O2 src/matrix_mpi.cpp -o src/matrix_mpi_cpp
fi

# 3. Compile CUDA (C / C++) if nvcc available
if command -v nvcc &> /dev/null; then
    echo "[+] Compiling CUDA (C/C++) implementations..."
    nvcc -O2 src/matrix_cuda.cu -o src/matrix_cuda_cu
    nvcc -O2 src/matrix_cuda.cpp -o src/matrix_cuda_cpp
fi

echo "=========================================================="
echo " Build & Verification Complete for C and C++ Source Codes."
echo "=========================================================="
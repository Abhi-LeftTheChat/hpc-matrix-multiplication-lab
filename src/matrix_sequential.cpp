#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

#define N 4000

int main() {
    std::cout << "Initializing " << N << " x " << N << " matrices..." << std::endl;

    double* A = new double[N * N];
    double* B = new double[N * N];
    double* C = new double[N * N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = 1.0;
            B[i * N + j] = 1.0;
            C[i * N + j] = 0.0;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "\nSequential C++ Matrix Multiplication Completed" << std::endl;
    std::cout << "Matrix Size = " << N << " x " << N << std::endl;
    std::cout << "Execution Time = " << std::fixed << std::setprecision(6) << duration.count() << " seconds" << std::endl;
    std::cout << "Verification C[0][0] = " << std::fixed << std::setprecision(2) << C[0] << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
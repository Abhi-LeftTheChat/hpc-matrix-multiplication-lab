#include <iostream>
#include <vector>
#include <iomanip>
#include <mpi.h>
#include <unistd.h>

#define N 4000

int main(int argc, char *argv[]) {
    int rank, size;
    char hostname[256];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    gethostname(hostname, sizeof(hostname));

    if (N % size != 0) {
        if (rank == 0) {
            std::cout << "Matrix size must be divisible by number of processes." << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    int rows_per_process = N / size;

    double *A = nullptr;
    double *B = new double[N * N];
    double *C = nullptr;

    double *local_A = new double[rows_per_process * N];
    double *local_C = new double[rows_per_process * N];

    if (rank == 0) {
        A = new double[N * N];
        C = new double[N * N];

        std::cout << "Initializing " << N << " x " << N << " matrices..." << std::endl;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = 1.0;
                B[i * N + j] = 1.0;
                C[i * N + j] = 0.0;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatter(A, rows_per_process * N, MPI_DOUBLE,
                local_A, rows_per_process * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::cout << "Rank " << rank << " on " << hostname << " computing " << rows_per_process << " rows" << std::endl;

    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i * N + j] = 0.0;
            for (int k = 0; k < N; k++) {
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
            }
        }
    }

    MPI_Gather(local_C, rows_per_process * N, MPI_DOUBLE,
               C, rows_per_process * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0) {
        std::cout << "\nMPI C++ Matrix Multiplication Completed" << std::endl;
        std::cout << "Matrix Size = " << N << " x " << N << std::endl;
        std::cout << "Number of MPI Processes = " << size << std::endl;
        std::cout << "Execution Time = " << std::fixed << std::setprecision(6) << (end - start) << " seconds" << std::endl;
        std::cout << "Verification C[0][0] = " << std::fixed << std::setprecision(2) << C[0] << std::endl;

        delete[] A;
        delete[] C;
    }

    delete[] B;
    delete[] local_A;
    delete[] local_C;

    MPI_Finalize();
    return 0;
}
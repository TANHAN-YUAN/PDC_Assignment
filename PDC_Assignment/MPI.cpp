#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <mpi.h>
#include <iomanip>

using namespace std;
// ---------------------------------------------------------
// MPI VERSION (Independent Function)
// ---------------------------------------------------------
double luMPI(const vector<vector<double>>& A_in,
    vector<vector<double>>& L_out,
    vector<vector<double>>& U_out,
    int n, int rank, int size)
{
    // Start Timer (Wall clock)
    double start_time = MPI_Wtime();

    int rows_per_proc = n / size;
    int remainder = n % size;

    if (remainder != 0 && rank == 0) {
        cerr << "[MPI Error] For this example, N must be divisible by the number of processes.\n";
        return 0.0;
    }

    // 1. Flatten 2D Vector to 1D Array (Rank 0 only)
    vector<double> flatA;
    if (rank == 0) {
        flatA.resize(n * n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                flatA[i * n + j] = A_in[i][j];
            }
        }
    }
    //[][][][]
    //[][][][] --> [][][][][][][][][][][][]
    //[][][][]

    // 2. Scatter Rows
    vector<double> localA(rows_per_proc * n);
    MPI_Scatter(flatA.data(), rows_per_proc * n, MPI_DOUBLE,
        localA.data(), rows_per_proc * n, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    vector<double> pivot_row(n);

    // 3. Parallel Gaussian Elimination
    for (int k = 0; k < n; k++) {
        int pivot_owner = k / rows_per_proc;

        if (rank == pivot_owner) {
            int local_index = (k % rows_per_proc) * n;
            for (int j = 0; j < n; j++) {
                pivot_row[j] = localA[local_index + j];
            }
        }

        MPI_Bcast(pivot_row.data(), n, MPI_DOUBLE, pivot_owner, MPI_COMM_WORLD);

        for (int i = 0; i < rows_per_proc; i++) {
            int global_row_idx = rank * rows_per_proc + i;

            if (global_row_idx > k) {
                int local_idx = i * n;
                double scale = localA[local_idx + k] / pivot_row[k];
                localA[local_idx + k] = scale;

                for (int j = k + 1; j < n; j++) {
                    localA[local_idx + j] -= scale * pivot_row[j];
                }
            }
        }
    }

    // 4. Gather results
    MPI_Gather(localA.data(), rows_per_proc * n, MPI_DOUBLE,
        flatA.data(), rows_per_proc * n, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    double time = end_time - start_time;

    // 5. Reconstruct L and U (Rank 0 only)
    if (rank == 0) {
        cout << "[MPI] Time taken: " << (time) << " seconds.\n";

        // Unpack flatA into L and U
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double val = flatA[i * n + j];
                if (i > j) {
                    L_out[i][j] = val;
                    U_out[i][j] = 0.0;
                }
                else if (i == j) {
                    L_out[i][j] = 1.0;
                    U_out[i][j] = val;
                }
                else {
                    L_out[i][j] = 0.0;
                    U_out[i][j] = val;
                }
            }
        }

        if (n <= 5) {
            cout << "L matrix (MPI):\n";
            for (const auto& row : L_out) {
                for (double val : row) cout << setw(8) << val << " ";
                cout << endl;
            }
            cout << "U matrix (MPI):\n";
            for (const auto& row : U_out) {
                for (double val : row) cout << setw(8) << val << " ";
                cout << endl;
            }
        }
    }

    return time;
}

double luMPI_CA(const vector<vector<double>>& A_in,
    vector<vector<double>>& L_out,
    vector<vector<double>>& U_out,
    int n, int rank, int size)
{
    // ==============================
    // Start wall-clock timer
    // ==============================
    double start_time = MPI_Wtime();

    // Panel width (controls communication vs computation)
    const int b = 32;

    int rows_per_proc = n / size;

    // ==============================
    // Input validation
    // ==============================
    if (n % size != 0 && rank == 0) {
        cerr << "[MPI Error] N must be divisible by number of processes.\n";
        return 0.0;
    }

    // ==============================
    // Flatten matrix on rank 0
    // ==============================
    vector<double> flatA;
    if (rank == 0) {
        flatA.resize(n * n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                flatA[i * n + j] = A_in[i][j];
    }

    // ==============================
    // Scatter rows to processes
    // ==============================
    vector<double> localA(rows_per_proc * n);
    MPI_Scatter(flatA.data(), rows_per_proc * n, MPI_DOUBLE,
        localA.data(), rows_per_proc * n, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    // ==============================
    // Buffer for panel broadcast
    // ==============================
    vector<double> panel(b * n);

    // ==============================
    // CA-LU Factorization
    // ==============================
    for (int k = 0; k < n; k += b) {

        // End column of current panel
        int panel_end = min(k + b, n);

        // Process that owns the panel rows
        int owner = k / rows_per_proc;

        // ==========================
        // Panel Factorization (Owner)
        // ==========================
        if (rank == owner) {

            // Factorize panel locally
            for (int i = k; i < panel_end; i++) {
                int local_i = (i % rows_per_proc) * n;

                // Eliminate previous columns in panel
                for (int j = k; j < i; j++) {

                    // Compute L(i,j)
                    localA[local_i + j] /=
                        localA[(j % rows_per_proc) * n + j];

                    // Update U(i,j+1 : panel_end)
                    for (int col = j + 1; col < panel_end; col++) {
                        localA[local_i + col] -=
                            localA[local_i + j] *
                            localA[(j % rows_per_proc) * n + col];
                    }
                }
            }

            // Copy panel rows into contiguous buffer
            for (int i = k; i < panel_end; i++) {
                int local_i = (i % rows_per_proc) * n;
                for (int j = 0; j < n; j++) {
                    panel[(i - k) * n + j] = localA[local_i + j];
                }
            }
        }

        // ==========================
        // Broadcast panel once
        // ==========================
        MPI_Bcast(panel.data(),
            (panel_end - k) * n,
            MPI_DOUBLE,
            owner,
            MPI_COMM_WORLD);

        // ==========================
        // Trailing matrix update
        // ==========================
        for (int i = 0; i < rows_per_proc; i++) {
            int global_i = rank * rows_per_proc + i;

            // Only rows below the panel
            if (global_i >= panel_end) {

                int local_idx = i * n;

                for (int p = k; p < panel_end; p++) {

                    // Compute L(i,p)
                    double scale =
                        localA[local_idx + p] /
                        panel[(p - k) * n + p];

                    localA[local_idx + p] = scale;

                    // Update U(i, panel_end:n)
                    for (int j = panel_end; j < n; j++) {
                        localA[local_idx + j] -=
                            scale * panel[(p - k) * n + j];
                    }
                }
            }
        }
    }

    // ==============================
    // Gather final matrix
    // ==============================
    MPI_Gather(localA.data(), rows_per_proc * n, MPI_DOUBLE,
        flatA.data(), rows_per_proc * n, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    double time = MPI_Wtime() - start_time;

    // ==============================
    // Reconstruct L and U (rank 0)
    // ==============================
    if (rank == 0) {
        cout << "[CA-LU MPI] Time taken: " << time << " seconds.\n";

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double val = flatA[i * n + j];

                if (i > j) {
                    L_out[i][j] = val;
                    U_out[i][j] = 0.0;
                }
                else if (i == j) {
                    L_out[i][j] = 1.0;
                    U_out[i][j] = val;
                }
                else {
                    L_out[i][j] = 0.0;
                    U_out[i][j] = val;
                }
            }
        }
    }

    return time;
}
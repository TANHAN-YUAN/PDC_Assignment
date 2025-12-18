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

double luMPI_CA(
    const vector<vector<double>>& A_in,
    vector<vector<double>>& L_out,
    vector<vector<double>>& U_out,
    int n, int rank, int size)
{
    const int b = 8;                 // panel width (small = more stable)
    const double EPS = 1e-12;        // pivot safety threshold

    double start_time = MPI_Wtime();

    // -----------------------------
    // Validate MPI layout
    // -----------------------------
    if (n % size != 0) {
        if (rank == 0)
            cerr << "[CA-LU Error] Matrix size must be divisible by MPI size\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rows_per_proc = n / size;

    // -----------------------------
    // Flatten matrix on rank 0
    // -----------------------------
    vector<double> flatA;
    if (rank == 0) {
        flatA.resize(n * n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                flatA[i * n + j] = A_in[i][j];
    }

    // -----------------------------
    // Scatter rows
    // -----------------------------
    vector<double> localA(rows_per_proc * n);
    MPI_Scatter(
        flatA.data(),
        rows_per_proc * n,
        MPI_DOUBLE,
        localA.data(),
        rows_per_proc * n,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // Buffer to broadcast panels
    vector<double> panel(b * n);

    // =============================
    // CA-LU Factorization
    // =============================
    for (int k = 0; k < n; k += b) {

        int panel_end = min(k + b, n);

        // Which rank owns the pivot rows?
        int owner = k / rows_per_proc;

        // -----------------------------
        // Panel factorization (owner)
        // -----------------------------
        if (rank == owner) {

            for (int i = k; i < panel_end; i++) {

                int local_i = (i % rows_per_proc) * n;

                // Pivot check
                double pivot = localA[local_i + i];
                if (fabs(pivot) < EPS) {
                    cerr << "[CA-LU Error] Zero pivot at row " << i << endl;
                    MPI_Abort(MPI_COMM_WORLD, 2);
                }

                // Compute multipliers
                for (int r = i + 1; r < panel_end; r++) {
                    int local_r = (r % rows_per_proc) * n;
                    localA[local_r + i] /= pivot;

                    // Update U entries inside panel
                    for (int c = i + 1; c < panel_end; c++) {
                        localA[local_r + c] -=
                            localA[local_r + i] * localA[local_i + c];
                    }
                }
            }

            // Copy panel rows into buffer
            for (int i = k; i < panel_end; i++) {
                int local_i = (i % rows_per_proc) * n;
                for (int j = 0; j < n; j++) {
                    panel[(i - k) * n + j] = localA[local_i + j];
                }
            }
        }

        // -----------------------------
        // Broadcast panel
        // -----------------------------
        MPI_Bcast(
            panel.data(),
            (panel_end - k) * n,
            MPI_DOUBLE,
            owner,
            MPI_COMM_WORLD
        );

        // -----------------------------
        // Trailing matrix update
        // -----------------------------
        for (int i = 0; i < rows_per_proc; i++) {

            int global_i = rank * rows_per_proc + i;
            if (global_i < panel_end) continue;

            int local_idx = i * n;

            for (int p = k; p < panel_end; p++) {

                double pivot = panel[(p - k) * n + p];
                if (fabs(pivot) < EPS) {
                    cerr << "[CA-LU Error] Zero pivot broadcast at " << p << endl;
                    MPI_Abort(MPI_COMM_WORLD, 3);
                }

                double multiplier =
                    localA[local_idx + p] / pivot;

                localA[local_idx + p] = multiplier;

                for (int j = panel_end; j < n; j++) {
                    localA[local_idx + j] -=
                        multiplier * panel[(p - k) * n + j];
                }
            }
        }
    }

    // =============================
    // Gather final matrix
    // =============================
    MPI_Gather(
        localA.data(),
        rows_per_proc * n,
        MPI_DOUBLE,
        flatA.data(),
        rows_per_proc * n,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    double time = MPI_Wtime() - start_time;

    // =============================
    // Reconstruct L and U (rank 0)
    // =============================
    if (rank == 0) {
        cout << "[CA-LU MPI] Time taken: " << time << " seconds\n";

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


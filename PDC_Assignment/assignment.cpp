#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <mpi.h>
#include <iomanip>

using namespace std;

// ---------------------------------------------------------
// SERIAL VERSION
// ---------------------------------------------------------
void luSerial(const vector<vector<double>>& A,
    vector<vector<double>>& L,
    vector<vector<double>>& U)
{
    int n = A.size();

    // Start Timer
    auto start = chrono::high_resolution_clock::now();

    // Initialize L and U
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            U[i][j] = 0.0;
            L[i][j] = 0.0;
        }
        L[i][i] = 1.0;
    }

    // Serial LU decomposition
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i <= j; i++)
        {
            double sum = 0.0;
            for (int k = 0; k < i; k++)
                sum += L[i][k] * U[k][j];

            U[i][j] = A[i][j] - sum;
        }

        for (int i = j + 1; i < n; i++)
        {
            double sum = 0.0;
            for (int k = 0; k < j; k++)
                sum += L[i][k] * U[k][j];

            L[i][j] = (A[i][j] - sum) / U[j][j];
        }
    }

    // Stop Timer
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "[Serial] Time taken: " << elapsed.count() << " seconds.\n";

    // Only print matrices if N is small (to avoid clutter)
    if (n <= 5) {
        cout << "L matrix (Serial):\n";
        for (auto& row : L) {
            for (double val : row) cout << setw(8) << val << " ";
            cout << endl;
        }
        cout << "U matrix (Serial):\n";
        for (auto& row : U) {
            for (double val : row) cout << setw(8) << val << " ";
            cout << endl;
        }
    }
}

// ---------------------------------------------------------
// MPI VERSION (Independent Function)
// ---------------------------------------------------------
void luMPI(const vector<vector<double>>& A_in,
    vector<vector<double>>& L_out,
    vector<vector<double>>& U_out,
    int n, int rank, int size)
{
    // Start Timer (Wall clock)
    double start_time = MPI_Wtime();

    int rows_per_proc = n / size;
    int remainder = n % size;

    // Safety check for this example code
    if (remainder != 0 && rank == 0) {
        cerr << "[MPI Error] For this example, N must be divisible by the number of processes.\n";
        return;
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

    // 2. Scatter Rows to processes
    // Each process gets 'rows_per_proc' rows. Total elements = rows_per_proc * n
    vector<double> localA(rows_per_proc * n);
    MPI_Scatter(flatA.data(), rows_per_proc * n, MPI_DOUBLE,
        localA.data(), rows_per_proc * n, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    // Buffer for the current pivot row
    vector<double> pivot_row(n);

    // 3. Parallel Gaussian Elimination
    for (int k = 0; k < n; k++) {

        // Determine who owns row k
        int pivot_owner = k / rows_per_proc;

        // If I own the pivot row, copy it to buffer
        if (rank == pivot_owner) {
            int local_index = (k % rows_per_proc) * n;
            for (int j = 0; j < n; j++) {
                pivot_row[j] = localA[local_index + j];
            }
        }

        // Broadcast pivot row to everyone
        MPI_Bcast(pivot_row.data(), n, MPI_DOUBLE, pivot_owner, MPI_COMM_WORLD);

        // Update local rows
        for (int i = 0; i < rows_per_proc; i++) {
            int global_row_idx = rank * rows_per_proc + i;

            // Only eliminate rows BELOW the pivot
            if (global_row_idx > k) {
                int local_idx = i * n;
                double scale = localA[local_idx + k] / pivot_row[k];

                // Store L value in the lower triangle of A
                localA[local_idx + k] = scale;

                // Update U values for the rest of the row
                for (int j = k + 1; j < n; j++) {
                    localA[local_idx + j] -= scale * pivot_row[j];
                }
            }
        }
    }

    // 4. Gather results back to Rank 0
    MPI_Gather(localA.data(), rows_per_proc * n, MPI_DOUBLE,
        flatA.data(), rows_per_proc * n, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    // Stop Timer
    double end_time = MPI_Wtime();

    // 5. Reconstruct L and U (Rank 0 only) and Print
    if (rank == 0) {
        cout << "[MPI] Time taken: " << (end_time - start_time) << " seconds.\n";

        // Unpack flatA into L and U
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double val = flatA[i * n + j];
                if (i > j) {
                    L_out[i][j] = val; // Lower part stores L
                    U_out[i][j] = 0.0;
                }
                else if (i == j) {
                    L_out[i][j] = 1.0; // Diagonal L is 1
                    U_out[i][j] = val; // Diagonal U
                }
                else {
                    L_out[i][j] = 0.0;
                    U_out[i][j] = val; // Upper part stores U
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
}

// ---------------------------------------------------------
// UTILS
// ---------------------------------------------------------
vector<vector<double>> generateVector(int n) {
    vector<vector<double>> A(n, vector<double>(n));
    // Use fixed seed for reproducibility across runs if needed
    // But typically only Rank 0 generates, so srand(time(0)) is fine.
    srand(time(0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = rand() % 10 + 1;
        }
    }
    return A;
}

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int vecSize = 500; // Use 4 so it's divisible by 2 or 4 processes
    vector<vector<double>> A;

    // Output matrices containers
    vector<vector<double>> L_serial(vecSize, vector<double>(vecSize));
    vector<vector<double>> U_serial(vecSize, vector<double>(vecSize));
    vector<vector<double>> L_mpi(vecSize, vector<double>(vecSize));
    vector<vector<double>> U_mpi(vecSize, vector<double>(vecSize));

    // 1. Generate Data (Rank 0 only)
    if (rank == 0) {
        A = generateVector(vecSize);
        cout << "--- Processing Matrix Size " << vecSize << " ---\n";

        // Run Serial on Rank 0
        cout << "\nRunning Serial Version...\n";
        luSerial(A, L_serial, U_serial);
    }

    // 2. Run MPI Version
    // We need a barrier to ensure printed text doesn't overlap
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) cout << "\nRunning MPI Version on " << size << " processes...\n";

    // Note: 'A' is empty on non-root ranks, but that's handled inside luMPI
    luMPI(A, L_mpi, U_mpi, vecSize, rank, size);

    MPI_Finalize();
    return 0;
}
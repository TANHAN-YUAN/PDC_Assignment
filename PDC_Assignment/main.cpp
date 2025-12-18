#include "LUHelpers.h"
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <iomanip>

using namespace std;

vector<vector<double>> generateVector(int n, bool safe_data) {
    vector<vector<double>> A(n, vector<double>(n));

    srand(time(nullptr));

    if (!safe_data) {
        // Original random matrix (NOT safe for no-pivot LU)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = rand() % 10;
            }
        }
        return A;
    }

    // ==============================
    // SAFE DATA: Diagonally dominant
    // ==============================
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;

        for (int j = 0; j < n; j++) {
            if (i != j) {
                A[i][j] = (rand() % 5) + 1;  // small off-diagonal values
                row_sum += abs(A[i][j]);
            }
        }

        // Strong diagonal dominance
        A[i][i] = row_sum + (rand() % 5) + 1;
    }

    return A;
}

double relativeResidual(const vector<vector<double>>& A,
    const vector<vector<double>>& L,
    const vector<vector<double>>& U)
{
    int n = A.size();
    double normA = 0.0;
    double normR = 0.0;

    // Compute Frobenius norm of A and residual (A - LU)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {

            // Compute (LU)[i][j]
            double LUij = 0.0;
            for (int k = 0; k < n; k++) {
                LUij += L[i][k] * U[k][j];
            }

            double diff = A[i][j] - LUij;

            normA += A[i][j] * A[i][j];
            normR += diff * diff;
        }
    }

    return sqrt(normR) / sqrt(normA);
}

double residualPivoted(const vector<vector<double>>& A,
    const vector<vector<double>>& L,
    const vector<vector<double>>& U,
    const vector<int>& P)
{
    int n = A.size();
    vector<vector<double>> PA(n, vector<double>(n));
    // Reorder A based on permutation vector P
    for (int i = 0; i < n; i++) {
        PA[i] = A[P[i]];
    }
    return relativeResidual(PA, L, U);
}


int main(int argc, char** argv)
{
    // ============================
    // Initialize timers
    // ============================
    double serial = 0.0;
    double omp = 0.0;
    double omp_basic = 0.0;
    double omp_piv = 0.0;
    double mpi = 0.0;
    double ca = 0.0;

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int vecSize;
    int numThreads;
    bool safe_data = false;
    char safeInput;

    // ============================
    // User Input (Rank 0 only)
    // ============================
    if (rank == 0) {
        cout << "Enter matrix size : ";
        cin >> vecSize;
        cout << "Enable safe data (y/n)? ";
        cin >> safeInput;
        safe_data = (safeInput == 'y' || safeInput == 'Y');
        cout << "Enter number of OpenMP threads: ";
        cin >> numThreads;

        // Set the number of threads for OpenMP
        omp_set_num_threads(numThreads);
    }


    MPI_Bcast(&vecSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    vector<vector<double>> A;

    // ============================
    // Output matrices
    // ============================


    vector<vector<double>> L_serial(vecSize, vector<double>(vecSize));
    vector<vector<double>> U_serial(vecSize, vector<double>(vecSize));

    vector<vector<double>> L_omp(vecSize, vector<double>(vecSize));
    vector<vector<double>> U_omp(vecSize, vector<double>(vecSize));

    vector<vector<double>> L_omp_piv(vecSize, vector<double>(vecSize));
    vector<vector<double>> U_omp_piv(vecSize, vector<double>(vecSize));
    vector<int> P; 

    vector<vector<double>> L_mpi(vecSize, vector<double>(vecSize));
    vector<vector<double>> U_mpi(vecSize, vector<double>(vecSize));

    vector<vector<double>> L_ca(vecSize, vector<double>(vecSize));
    vector<vector<double>> U_ca(vecSize, vector<double>(vecSize));

    // ============================
    // Generate data (rank 0)
    // ============================
    if (rank == 0) {
        A = generateVector(vecSize, safe_data);
        
        cout << "====================================================\n";
        cout << "LU Decomposition Benchmark | Matrix Size: " << vecSize << "x" << vecSize << "\n";
        cout << "MPI Processes: " << size << "OpenMP Threads: " << omp_get_max_threads() << "\n";

        if (vecSize <= 5) {
             cout << "Matrix A:" << endl;
            for (int i = 0; i < vecSize; i++) {
                for (int j = 0; j < vecSize; j++) {
                    cout << A[i][j] << " ";
                }
                cout << endl;
            }
        }

        cout << "====================================================\n";

        cout << "\nRunning Serial Version...\n";
        serial = luSerial(A, L_serial, U_serial);


        cout << "\nRunning OpenMP Version...\n";
        omp_basic = luOMP(A, L_omp, U_omp);
      

        cout << "\nRunning OpenMP (Pivoting) Version...\n";
        omp_piv = luOMP_Pivoting(A, L_omp_piv, U_omp_piv, P);

    }
   
    // ============================
    // Standard MPI LU
    // ============================
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        cout << "\nRunning MPI LU Version on "
        << size << " processes...\n";

    mpi = luMPI(A, L_mpi, U_mpi, vecSize, rank, size);

    // ============================
    // CA-LU MPI
    // ============================
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        cout << "\nRunning CA-LU MPI Version on "
        << size << " processes...\n";

    ca = luMPI_CA(A, L_ca, U_ca, vecSize, rank, size);
 
    MPI_Finalize();
    // ============================
    // Final comparison (rank 0)
    // ============================
    if (rank == 0) {
        cout << "\nRunning Relative Residual......" << endl;
        // ---- Relative Residuals ----
        double res_serial = relativeResidual(A, L_serial, U_serial);
        double res_omp = relativeResidual(A, L_omp, U_omp);
        double res_omp_piv = residualPivoted(A, L_omp_piv, U_omp_piv, P);
        double res_mpi = relativeResidual(A, L_mpi, U_mpi);
        double res_ca = relativeResidual(A, L_ca, U_ca);

        cout << "\n--- Final Results ---\n";

        cout << fixed << setprecision(6);
        cout << "Serial LU Time: " << serial << " s\n";
        cout << "OpenMP Basic Time:   " << omp_basic << " s\n";
        cout << "OpenMP Pivoting Time: " << omp_piv << " s\n";
        cout << "MPI LU Time:    " << mpi << " s\n";
        cout << "CA-LU Time:     " << ca << " s\n";

        cout << "\n--- Relative Residuals ---\n";
        cout << scientific << setprecision(3);
        cout << "Serial LU Residual: " << res_serial << "\n";
        cout << "OpenMP Basic Residual: " << res_omp << "\n";
        cout << "OpenMP Piv Residual:  " << res_omp_piv << "\n";
        cout << "MPI LU Residual:    " << res_mpi << "\n";
        cout << "CA-LU Residual:     " << res_ca << "\n";

        // ---- Speedup ----
        cout << "\n--- Speedup (vs Serial) ---\n";
        cout << "OpenMP Basic Speedup:  " << serial / omp_basic << "x\n";
        cout << "OpenMP Piv Speedup:    " << serial / omp_piv << "x\n";
        cout << "MPI LU Speedup:    " << serial / mpi << "\n";
        cout << "CA-LU Speedup:     " << serial / ca << "\n";

        // ---- Efficiency ----
        cout << "\n--- Efficiency ---\n";

        cout << "OpenMP Basic Efficiency: " << (serial / omp_basic) / omp_get_max_threads() << "\n";
        cout << "OpenMP Piv Efficiency:   " << (serial / omp_piv) / omp_get_max_threads() << "\n";
        cout << "MPI LU Efficiency:    "
            << (serial / mpi) / size << "\n";
        cout << "CA-LU Efficiency:     "
            << (serial / ca) / size << "\n";

        // ---- MPI vs CA-LU ----
        cout << "\n--- MPI vs CA-LU Comparison ---\n";
        cout << "CA-LU speedup over MPI: "
            << mpi / ca << "x\n";
    }

    return 0;
}

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

vector<vector<double>> generateVector(int n) {
    vector<vector<double>> A(n, vector<double>(n));
    // Use fixed seed for reproducibility across runs if needed
    // But typically only Rank 0 generates, so srand(time(0)) is fine.
    srand(time(0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (rand() % 10) + 1;
        }
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


int main(int argc, char** argv)
{
    // ============================
    // Initialize timers
    // ============================
    double serial = 0.0;
    double omp = 0.0;

    double mpi = 0.0;
    double ca = 0.0;

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int vecSize = 500;

    vector<vector<double>> A;

    // ============================
    // Output matrices
    // ============================


    vector<vector<double>> L_serial(vecSize, vector<double>(vecSize));
    vector<vector<double>> U_serial(vecSize, vector<double>(vecSize));

    vector<vector<double>> L_mpi(vecSize, vector<double>(vecSize));
    vector<vector<double>> U_mpi(vecSize, vector<double>(vecSize));

    vector<vector<double>> L_ca(vecSize, vector<double>(vecSize));
    vector<vector<double>> U_ca(vecSize, vector<double>(vecSize));

    // ============================
    // Generate data (rank 0)
    // ============================
    if (rank == 0) {
        A = generateVector(vecSize);

        cout << "====================================================\n";
        cout << "LU Decomposition Benchmark | Matrix Size: " << vecSize << "x" << vecSize << "\n";
        cout << "MPI Processes: " << size << "\n";
        cout << "====================================================\n";

        cout << "\nRunning Serial Version...\n";
        serial = luSerial(A, L_serial, U_serial);


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
 

    vector<vector<double>> L_omp(vecSize, vector<double>(vecSize));
    vector<vector<double>> U_omp(vecSize, vector<double>(vecSize));

    // ============================
    // Final comparison (rank 0)
    // ============================
    if (rank == 0) {

        // ---- Relative Residuals ----
        double res_serial = relativeResidual(A, L_serial, U_serial);
        double res_omp = relativeResidual(A, L_omp, U_omp);
        double res_mpi = relativeResidual(A, L_mpi, U_mpi);
        double res_ca = relativeResidual(A, L_ca, U_ca);

        cout << "\n--- Final Results & Relative Residuals ---\n";

        cout << fixed << setprecision(6);
        cout << "Serial LU Time: " << serial << " s\n";
        cout << "OMP LU Time: " << omp << " s\n";
        cout << "MPI LU Time:    " << mpi << " s\n";
        cout << "CA-LU Time:     " << ca << " s\n";

        cout << scientific << setprecision(3);
        cout << "Serial LU Residual: " << res_serial << "\n";
        cout << "Serial LU Residual: " << res_serial << "\n";
        cout << "MPI LU Residual:    " << res_mpi << "\n";
        cout << "CA-LU Residual:     " << res_ca << "\n";

        // ---- Speedup ----
        cout << "\n--- Speedup (vs Serial) ---\n";
        cout << "MPI LU Speedup:    " << serial / mpi << "\n";
        cout << "CA-LU Speedup:     " << serial / ca << "\n";

        // ---- Efficiency ----
        cout << "\n--- Efficiency ---\n";
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

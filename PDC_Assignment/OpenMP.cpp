#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>


using namespace std;

// ---------------------------------------------------------
// OpenMP VERSION
// ---------------------------------------------------------
// 1. Basic Parallel LU (Doolittle) - Fast but no pivoting
double luOMP(
    const vector<vector<double>>& A,
    vector<vector<double>>& L,
    vector<vector<double>>& U
) {
    int n = A.size();
    U = A; // Initialize U as A

    // Initialize L as Identity Matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    double start = omp_get_wtime();

    for (int k = 0; k < n; k++) {
        // Parallelize the row updates
#pragma omp parallel for schedule(static)
        for (int i = k + 1; i < n; i++) {
            double factor = U[i][k] / U[k][k];
            L[i][k] = factor;
            for (int j = k; j < n; j++) {
                U[i][j] -= factor * U[k][j];
            }
        }
    }
    


    double end = omp_get_wtime() - start;
    cout << "[OpenMP Basic] Time taken: " << end << " seconds." << endl; 
    if (n <= 5) {
        cout << "L matrix (OpenMP Basic):\n";
        for (const auto& row : L) {
            for (double val : row)
                cout << setw(8) << fixed << setprecision(2) << val << " ";
            cout << endl;
        }

        cout << "U matrix (OpenMP Basic):\n";
        for (const auto& row : U) {
            for (double val : row)
                cout << setw(8) << fixed << setprecision(2) << val << " ";
            cout << endl;
        }
        cout << "====================================================\n";
    }

    return end;
}


double luOMP_Pivoting(const vector<vector<double>>& A, vector<vector<double>>& L, vector<vector<double>>& U, vector<int>& P) {
    int n = A.size();
    U = A;
    P.resize(n);
    for (int i = 0; i < n; i++) P[i] = i; // Initial permutation

    // Initialize L
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) L[i][j] = (i == j) ? 1.0 : 0.0;
    }

    double start = omp_get_wtime();

    for (int k = 0; k < n; k++) {
        // --- Step 1: Partial Pivoting (Serial Search is safer) ---
        int maxRow = k;
        double maxVal = abs(U[k][k]);
        for (int i = k + 1; i < n; i++) {
            if (abs(U[i][k]) > maxVal) {
                maxVal = abs(U[i][k]);
                maxRow = i;
            }
        }

        // Swap rows in U, L (multipliers), and P (record)
        if (maxRow != k) {
            swap(U[k], U[maxRow]);
            swap(P[k], P[maxRow]);
            // Swap L elements already computed
            for (int j = 0; j < k; j++) swap(L[k][j], L[maxRow][j]);
        }

        // --- Step 2: Parallel Elimination ---
#pragma omp parallel for schedule(static)
        for (int i = k + 1; i < n; i++) {
            double factor = U[i][k] / U[k][k];
            L[i][k] = factor;
            for (int j = k; j < n; j++) {
                U[i][j] -= factor * U[k][j];
            }
        }
    }
    double end = omp_get_wtime() - start;



    cout << "[OpenMP Pivoting] Time taken: " << end << " seconds." << endl; 
    if (n <= 5) {
        cout << "L matrix (OpenMP Pivoting):\n";
        for (const auto& row : L) {
            for (double val : row)
                cout << setw(8) << fixed << setprecision(2) << val << " ";
            cout << endl;
        }

        cout << "U matrix (OpenMP Pivoting):\n";
        for (const auto& row : U) {
            for (double val : row)
                cout << setw(8) << fixed << setprecision(2) << val << " ";
            cout << endl;
        }

        cout << "Permutation vector P:\n";
        for (int p : P) cout << p << " ";
        cout << endl;
        cout << "====================================================\n";
    }

    return end;
}
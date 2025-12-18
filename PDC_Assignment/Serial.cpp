#include "LUHelpers.h"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace std;

// ---------------------------------------------------------
// SERIAL VERSION
// ---------------------------------------------------------
double luSerial(const vector<vector<double>>& A,
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

    return elapsed.count();
}
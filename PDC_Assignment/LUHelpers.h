#pragma once
#include <vector>

using namespace std;

// Serial LU
double luSerial(const vector<vector<double>>& A,
    vector<vector<double>>& L,
    vector<vector<double>>& U);

// OpenMP LU (Added these two)
double luOMP(const vector<vector<double>>& A,
    vector<vector<double>>& L,
    vector<vector<double>>& U);

double luOMP_Pivoting(const vector<vector<double>>& A,
    vector<vector<double>>& L,
    vector<vector<double>>& U,
    vector<int>& P);

// MPI LU
double luMPI(const vector<vector<double>>& A_in,
    vector<vector<double>>& L_out,
    vector<vector<double>>& U_out,
    int n, int rank, int size);

// CA-LU MPI
double luMPI_CA(const vector<vector<double>>& A_in,
    vector<vector<double>>& L_out,
    vector<vector<double>>& U_out,
    int n, int rank, int size);

// Helper functions
vector<vector<double>> generateVector(int n, bool safe_data );
double relativeResidual(const vector<vector<double>>& A,
    const vector<vector<double>>& L,
    const vector<vector<double>>& U);
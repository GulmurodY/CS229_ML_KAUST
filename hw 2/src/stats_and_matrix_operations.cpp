#include<iostream>
#include<cmath>
#include<vector>
#include<sstream>
#include<fstream>
#include<string>
#include<thread>
#include<map>  
#include<stdexcept>
#include "basis_functions.h"

double getMean(std::vector<double>& nums) 
{
    double sum = 0;
    for (double num: nums) {
        sum += num;
    }
    return sum / double(nums.size());
}

double getVariance(std::vector<double>& nums) 
{
    double x_mean = getMean(nums);
    int n = nums.size();
    double sum_deviation_x = 0;
    for (double x: nums) {
        double deviation_x = x - x_mean;
        sum_deviation_x += deviation_x * deviation_x;
    }
    return sum_deviation_x / double(n - 1);
}

double getStandardDeviation(std::vector<double>& nums) 
{
    double variance_x = getVariance(nums);
    return sqrt(variance_x);
}

std::vector<std::vector<double>> getTranspose(const std::vector<std::vector<double>>& mat) 
{
    int rows = mat.size();
    int cols = mat[0].size();
    
    std::vector<std::vector<double>> transposed(cols, std::vector<double>(rows));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transposed[j][i] = mat[i][j];
        }
    }
    
    return transposed;
}

void getModelMatrices(
    const std::vector<std::vector<double>>& data, 
    std::vector<std::vector<double>>& X, std::vector<double>& y) 
{
    size_t n = data.size();
    size_t m = data[0].size() - 1;

    X.resize(n, std::vector<double>(m + 1, 1.0)); 
    y.resize(n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            X[i][j + 1] = data[i][j];
        }
        y[i] = data[i][m];
    }
}


std::vector<std::vector<double>> getMatrixProduct(
    const std::vector<std::vector<double>>& A, 
    const std::vector<std::vector<double>>& B) 
{
    if (A.empty() || B.empty()) {
        throw std::invalid_argument("Matrices cannot be empty.");
    }
    
    size_t M = A.size();      
    size_t N = A[0].size();       
    size_t Q = B.size();    
    size_t P = B[0].size();
    
    if (N != Q) {
        throw std::invalid_argument("Matrix dimensions mismatch: A columns must equal B rows.");
    }

    std::vector<std::vector<double>> C(M, std::vector<double>(P, 0.0));

    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 2;
    
    std::vector<std::thread> threads;

    auto compute_rows = [&](size_t start_row, size_t end_row) {
        for (size_t i = start_row; i < end_row; ++i) {
            for (size_t j = 0; j < P; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < N; ++k) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
    };

    size_t rows_per_thread = (M + num_threads - 1) / num_threads;

    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start_row = t * rows_per_thread;
        size_t end_row = std::min(start_row + rows_per_thread, M);

        if (start_row < M) {
            threads.emplace_back(compute_rows, start_row, end_row);
        }
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return C;
}

std::vector<std::vector<double>> getMatrixInverse(const std::vector<std::vector<double>>& matrix) 
{
    size_t n = matrix.size();
    

    if (n == 0 || n != matrix[0].size()) {
        throw std::invalid_argument("Matrix must be square to be inverted.");
    }


    std::vector<std::vector<double>> aug(n, std::vector<double>(2 * n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            aug[i][j] = matrix[i][j];
        }
        aug[i][i + n] = 1.0; 
    }


    for (size_t i = 0; i < n; ++i) {

        size_t pivot = i;
        for (size_t j = i + 1; j < n; ++j) {
            if (std::abs(aug[j][i]) > std::abs(aug[pivot][i])) pivot = j;
        }
        std::swap(aug[i], aug[pivot]);


        if (std::abs(aug[i][i]) < 1e-10) {
            throw std::runtime_error("Matrix is singular and cannot be inverted.");
        }


        double factor = aug[i][i];
        for (size_t j = 0; j < 2 * n; ++j) {
            aug[i][j] /= factor;
        }


        for (size_t k = 0; k < n; ++k) {
            if (k != i) {
                double f = aug[k][i];
                for (size_t j = 0; j < 2 * n; ++j) {
                    aug[k][j] -= f * aug[i][j];
                }
            }
        }
    }


    std::vector<std::vector<double>> inverse(n, std::vector<double>(n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            inverse[i][j] = aug[i][j + n];
        }
    }

    return inverse;
}



#include<iostream>
#include<cmath>
#include<vector>
#include<sstream>
#include<fstream>
#include<string>
#include<thread>
#include <chrono>
#include <random>
#include "stats_and_matrix_operations.h"
#include "linear_regression_models.h"

DatasetSplit splitData(std::vector<std::vector<double>> data, double trainRatio) {
    std::random_device rd;
    std::default_random_engine rng(rd());
    std::shuffle(data.begin(), data.end(), rng);

    size_t trainSize = static_cast<size_t>(data.size() * trainRatio);
    
    DatasetSplit split;
    split.train.assign(data.begin(), data.begin() + trainSize);
    split.validation.assign(data.begin() + trainSize, data.end());

    return split;
}

std::vector<std::vector<double>> expandPolynomial(const std::vector<std::vector<double>>& data, int m) {
    if (m <= 1) return data;

    std::vector<std::vector<double>> expandedData;

    for (const auto& row : data) {
        std::vector<double> newRow;
        
        double target = row.back();

        for (size_t i = 0; i < row.size() - 1; ++i) {
            double originalFeature = row[i];
            
            for (int p = 1; p <= m; ++p) {
                newRow.push_back(std::pow(originalFeature, p));
            }
        }

        newRow.push_back(target);
        expandedData.push_back(newRow);
    }

    return expandedData;
}

std::vector<double> linearRegressionClosedForm(const std::vector<std::vector<double>>& data, double lambda) {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    getModelMatrices(data, X, y);
    
    size_t n = X.size();  
    size_t m = X[0].size(); 
    
    std::vector<std::vector<double>> X_T = getTranspose(X);
    
    std::vector<std::vector<double>> X_T_X = getMatrixProduct(X_T, X);


    for (size_t i = 1; i < m; ++i) {
        X_T_X[i][i] += lambda;
    }

    std::vector<std::vector<double>> X_T_X_inv = getMatrixInverse(X_T_X);
    
    std::vector<std::vector<double>> y_matrix(n, std::vector<double>(1));
    for (size_t i = 0; i < n; ++i) {
        y_matrix[i][0] = y[i];
    }
    
    std::vector<std::vector<double>> X_T_y = getMatrixProduct(X_T, y_matrix);
    
    std::vector<std::vector<double>> beta_matrix = getMatrixProduct(X_T_X_inv, X_T_y);
    
    std::vector<double> beta(m);
    for (size_t i = 0; i < m; ++i) {
        beta[i] = beta_matrix[i][0];
    }
    
    return beta;
}

std::vector<double> linearRegressionGradientDescent(const std::vector<std::vector<double>>& data,  double alpha,  int iterations, double lambda) {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    
    getModelMatrices(data, X, y);

    size_t n = X.size();          
    size_t m = X[0].size();       
    std::vector<double> beta(m, 0.0);
    
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<std::vector<double>> beta_mat(m, std::vector<double>(1));
        for(size_t i = 0; i < m; ++i) beta_mat[i][0] = beta[i];
        
        std::vector<std::vector<double>> pred_mat = getMatrixProduct(X, beta_mat);
        
        std::vector<std::vector<double>> error_mat(n, std::vector<double>(1));
        double total_mse = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double error = pred_mat[i][0] - y[i];
            error_mat[i][0] = error;
            total_mse += error * error;
        }
        if (iter % 50 == 0) {
            std::cout << " iteration " << iter << "current loss: " << total_mse << std::endl;
        }

        std::vector<std::vector<double>> XT = getTranspose(X);
        std::vector<std::vector<double>> grad_mat = getMatrixProduct(XT, error_mat);
        
        for (size_t j = 0; j < m; ++j) {
            double gradient = grad_mat[j][0] / n;
            
            if (j == 0) {
                beta[j] -= alpha * gradient;
            } else {
                beta[j] -= alpha * (gradient + (lambda / n));
            }
        }
    }
    
    return beta;
}

void compare_methods(const std::vector<std::vector<double>>& data) {
    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<double> w_cf = linearRegressionClosedForm(data);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cf_time = t2 - t1;

    auto t3 = std::chrono::high_resolution_clock::now();
    std::vector<double> w_gd = linearRegressionGradientDescent(data);
    auto t4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gd_time = t4 - t3;

    std::cout << "CF took: " << cf_time.count() << "s" << std::endl;
    std::cout << "GD took: " << gd_time.count() << "s" << std::endl;

    double abs_diff_sum = 0.0;
    for (int i = 0; i < w_cf.size(); ++i) {
        double d = w_cf[i] - w_gd[i];
        abs_diff_sum += std::abs(d);
        std::cout << "w[" << i << "] -> CF: " << w_cf[i] << ", GD: " << w_gd[i] << std::endl;
    }

    std::cout << "Total diff: " << abs_diff_sum << std::endl;
}
void checkEffectOfLambda(const std::vector<std::vector<double>>& dataset) {
    std::cout << "Closed Form solution" << std::endl;
    std::vector<double> lambdas = {0.0, 0.01, 0.1, 1.0, 10.0};
    for (double lambda : lambdas) {
        std::vector<double> weights = linearRegressionClosedForm(dataset, lambda);
        std::cout << "Lambda: " << lambda << " | Weights: ";
        for (double w : weights) {
            std::cout << w << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Gradient Descent solution" << std::endl; 
    for (double lambda : lambdas) {
        std::vector<double> weights = linearRegressionGradientDescent(dataset, 0.01, 1000, lambda);
        std::cout << "Lambda: " << lambda << " | Weights: ";
        for (double w : weights) {
            std::cout << w << " ";
        }
        std::cout << std::endl;
    }
}

double calculateMSE(const std::vector<std::vector<double>>& data, const std::vector<double>& weights) {
    double totalSquaredError = 0.0;
    size_t n = data.size();

    for (const auto& row : data) {
        double prediction = weights[0];
        
        for (size_t i = 0; i < row.size() - 1; ++i) {
            prediction += weights[i + 1] * row[i];
        }

        double actual = row.back();

        double error = prediction - actual;
        totalSquaredError += (error * error);
    }

    return totalSquaredError / static_cast<double>(n);
}

void saveResultsToCSV(std::string filename, std::vector<std::string> labels, std::vector<double> values) {
    std::ofstream file(filename);
    file << "Metric,Value\n"; 
    
    for(size_t i = 0; i < labels.size(); ++i) {
        file << labels[i] << "," << values[i] << "\n";
    }
    file.close();
}

void saveResultsToCSV(std::string filename, std::vector<double> x_axis, std::vector<double> train_mse, std::vector<double> val_mse) {
    std::ofstream file(filename);
    file << "X_Value,Training_MSE,Validation_MSE\n";
    
    for (size_t i = 0; i < x_axis.size(); ++i) {
        file << x_axis[i] << "," << train_mse[i] << "," << val_mse[i] << "\n";
    }
    file.close();
}
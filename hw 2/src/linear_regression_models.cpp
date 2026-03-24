#include<iostream>
#include<cmath>
#include<vector>
#include<fstream>
#include<string>
#include <random>
#include "stats_and_matrix_operations.h"
#include "linear_regression_models.h"
#include "basis_functions.h"

std::vector<std::vector<double>> buildDesignMatrix(
    const std::vector<std::vector<double>>& data,
    const BasisTransformConfig& config)
{
    std::vector<std::vector<double>> transformed;

    if (config.basisType == "polynomial") {
        transformed = applyPolynomialExpansion(data, config.polynomialDegree);
    } else {
        transformed = applyBasisForRegression(data, config);
    }

    std::vector<std::vector<double>> Phi;
    Phi.reserve(transformed.size());

    for (const auto& row : transformed) {
        std::vector<double> phi_row;
        phi_row.push_back(1.0);
        for (size_t j = 0; j + 1 < row.size(); ++j) {
            phi_row.push_back(row[j]);
        }
        Phi.push_back(std::move(phi_row));
    }

    return Phi;
}

BayesianLinearRegression computePosterior(
    const std::vector<std::vector<double>>& data,
    double alpha,
    double beta,
    const BasisTransformConfig& config)
{
    auto Phi = buildDesignMatrix(data, config);
    size_t N = Phi.size();
    size_t M = Phi[0].size();

    std::vector<double> t(N);
    for (size_t i = 0; i < N; ++i) t[i] = data[i].back();

    auto Phi_T = getTranspose(Phi);
    auto Phi_T_Phi = getMatrixProduct(Phi_T, Phi);

    std::vector<std::vector<double>> S_N_inv(M, std::vector<double>(M, 0.0));
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < M; ++j) {
            S_N_inv[i][j] = beta * Phi_T_Phi[i][j];
        }
        S_N_inv[i][i] += alpha;
    }

    auto S_N = getMatrixInverse(S_N_inv);

    std::vector<std::vector<double>> t_mat(N, std::vector<double>(1));
    for (size_t i = 0; i < N; ++i) t_mat[i][0] = t[i];

    auto Phi_T_t = getMatrixProduct(Phi_T, t_mat);
    auto S_N_Phi_T_t = getMatrixProduct(S_N, Phi_T_t);

    std::vector<double> m_N(M);
    for (size_t i = 0; i < M; ++i) m_N[i] = beta * S_N_Phi_T_t[i][0];

    BayesianLinearRegression model;
    model.m_N = m_N;
    model.S_N = S_N;
    model.alpha = alpha;
    model.beta = beta;
    model.basisConfig = config;
    return model;
}

double predictiveMean(
    const std::vector<double>& phi_x,
    const std::vector<double>& m_N)
{
    double mean = 0.0;
    for (size_t i = 0; i < m_N.size(); ++i) mean += m_N[i] * phi_x[i];
    return mean;
}

double predictiveVariance(
    const std::vector<double>& phi_x,
    const std::vector<std::vector<double>>& S_N,
    double beta)
{
    size_t M = phi_x.size();
    std::vector<double> S_N_phi(M, 0.0);
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < M; ++j)
            S_N_phi[i] += S_N[i][j] * phi_x[j];

    double quad = 0.0;
    for (size_t i = 0; i < M; ++i) quad += phi_x[i] * S_N_phi[i];

    return (1.0 / beta) + quad;
}

DatasetSplit splitData(
    std::vector<std::vector<double>> data, 
    double trainRatio) 
{
    std::random_device rd;
    std::default_random_engine rng(rd());
    std::shuffle(data.begin(), data.end(), rng);

    size_t trainSize = static_cast<size_t>(data.size() * trainRatio);
    
    DatasetSplit split;
    split.train.assign(data.begin(), data.begin() + trainSize);
    split.validation.assign(data.begin() + trainSize, data.end());

    return split;
}

std::vector<double> linearRegressionClosedForm(
    const std::vector<std::vector<double>>& data, 
    double lambda) 
{
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

double calculateMSE(
    const std::vector<std::vector<double>>& data, 
    const std::vector<double>& weights) 
{
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

void saveResultsToCSV(
    std::string filename, 
    std::vector<std::string> labels, 
    std::vector<double> values) 
{
    std::ofstream file(filename);
    file << "Metric,Value\n"; 
    
    for(size_t i = 0; i < labels.size(); ++i) {
        file << labels[i] << "," << values[i] << "\n";
    }
    file.close();
}

double bayesianMSE(const std::vector<std::vector<double>>& data,
                   const BayesianLinearRegression& model)
{
    auto Phi = buildDesignMatrix(data, model.basisConfig);
    double total = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        double err = data[i].back() - predictiveMean(Phi[i], model.m_N);
        total += err * err;
    }
    return total / data.size();
}

void savePosteriorToCSV(
    const std::string& filename,
    const BayesianLinearRegression& model)
{
    std::ofstream file(filename);
    file << "posterior_mean\n";
    for (size_t i = 0; i < model.m_N.size(); ++i) {
        file << "m_N[" << i << "]," << model.m_N[i] << "\n";
    }

    file << "\nposterior_covariance\n";
    for (size_t i = 0; i < model.S_N.size(); ++i) {
        for (size_t j = 0; j < model.S_N[i].size(); ++j) {
            file << model.S_N[i][j];
            if (j + 1 < model.S_N[i].size()) file << ",";
        }
        file << std::endl;
    }

    file.close();
}

void savePredictionsToCSV(
    const std::string& filename,
    const std::vector<std::vector<double>>& data,
    const BayesianLinearRegression& model)
{
    auto Phi = buildDesignMatrix(data, model.basisConfig);

    size_t nFeatures = data.empty() ? 0 : data[0].size() - 1;

    std::ofstream file(filename);
    for (size_t j = 0; j < nFeatures; ++j) file << "x" << j << ",";
    file << "actual, predicted_mean, lower_95CI, upper_95CI\n";

    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < nFeatures; ++j) file << data[i][j] << ",";

        double actual = data[i].back();
        double predicted_mean     = predictiveMean(Phi[i], model.m_N);
        double predicted_variance    = predictiveVariance(Phi[i], model.S_N, model.beta);
        double standard_deviatioin   = std::sqrt(predicted_variance);
        double lower_95CI     = predicted_mean - 1.96 * standard_deviatioin;
        double upper_95CI     = predicted_mean + 1.96 * standard_deviatioin;
        file << actual << ", " << predicted_mean << ", " << lower_95CI << ", " << upper_95CI << std::endl;
    }

    file.close();
}
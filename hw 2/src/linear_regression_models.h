#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include<vector>
#include<string>
#include "basis_functions.h"

struct DatasetSplit {
    std::vector<std::vector<double>> train;
    std::vector<std::vector<double>> validation;
};

std::vector<double> linearRegressionClosedForm(const std::vector<std::vector<double>>& data, double lambda = 0.1);
DatasetSplit splitData(std::vector<std::vector<double>> data, double trainRatio = 0.8);
double calculateMSE(const std::vector<std::vector<double>>& data, const std::vector<double>& weights);
void saveResultsToCSV(std::string filename, std::vector<std::string> labels, std::vector<double> values);


struct BayesianLinearRegression {
    std::vector<double> m_N;                    // posterior mean weights
    std::vector<std::vector<double>> S_N;       // posterior covariance matrix
    double alpha;                               // prior precision
    double beta;                                // noise precision (1/sigma^2)
    BasisTransformConfig basisConfig;
};

std::vector<std::vector<double>> buildDesignMatrix(
    const std::vector<std::vector<double>>& data,
    const BasisTransformConfig& config);

BayesianLinearRegression computePosterior(
    const std::vector<std::vector<double>>& data,
    double alpha,
    double beta,
    const BasisTransformConfig& config);

double predictiveMean(
    const std::vector<double>& phi_x,
    const std::vector<double>& m_N);

double predictiveVariance(
    const std::vector<double>& phi_x,
    const std::vector<std::vector<double>>& S_N,
    double beta);

double bayesianMSE(const std::vector<std::vector<double>>& data,
                   const BayesianLinearRegression& model);

void savePosteriorToCSV(
    const std::string& filename,
    const BayesianLinearRegression& model);

void savePredictionsToCSV(
    const std::string& filename,
    const std::vector<std::vector<double>>& data,
    const BayesianLinearRegression& model);

#endif

#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include<vector>

struct DatasetSplit {
    std::vector<std::vector<double>> train;
    std::vector<std::vector<double>> validation;
};

std::vector<double> linearRegressionClosedForm(const std::vector<std::vector<double>>& data, double lambda = 0.1); ;
std::vector<double> linearRegressionGradientDescent(const std::vector<std::vector<double>>& data,  double alpha = 0.0001,  int iterations = 2000, double lambda = 0.1);
void compare_methods(const std::vector<std::vector<double>>& dataset);
std::vector<std::vector<double>> expandPolynomial(const std::vector<std::vector<double>>& data, int m);
void checkEffectOfLambda(const std::vector<std::vector<double>>& dataset);
DatasetSplit splitData(std::vector<std::vector<double>> data, double trainRatio = 0.8);
double calculateMSE(const std::vector<std::vector<double>>& data, const std::vector<double>& weights);
void saveResultsToCSV(std::string filename, std::vector<std::string> labels, std::vector<double> values);
void saveResultsToCSV(std::string filename, std::vector<double> x_axis, std::vector<double> train_mse, std::vector<double> val_mse);

#endif 

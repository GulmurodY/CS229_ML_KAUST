#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <vector>

struct LogRegConfig {
    double learningRate = 0.1;
    int    epochs       = 1000;
    double l2           = 0.0;
    bool   verbose      = false;
};

struct LogRegModel {
    std::vector<double> weights;
    std::vector<double> lossHistory;
};

LogRegModel fitLogisticRegression(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    const LogRegConfig& cfg);

double predictProb(const std::vector<double>& x, const std::vector<double>& weights);
int    predictClass(const std::vector<double>& x, const std::vector<double>& weights, double threshold = 0.5);

double accuracy(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    const std::vector<double>& weights);

#endif

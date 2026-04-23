#include "logistic_regression.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

static double sigmoid(double z)
{
    if (z >= 0.0) {
        double e = std::exp(-z);
        return 1.0 / (1.0 + e);
    } else {
        double e = std::exp(z);
        return e / (1.0 + e);
    }
}

static double dot(const std::vector<double>& a, const std::vector<double>& b)
{
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        s += a[i] * b[i];
    }
    return s;
}

double predictProb(const std::vector<double>& x, const std::vector<double>& weights)
{
    return sigmoid(dot(x, weights));
}

int predictClass(const std::vector<double>& x, const std::vector<double>& weights, double threshold)
{
    return predictProb(x, weights) >= threshold ? 1 : 0;
}

LogRegModel fitLogisticRegression(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    const LogRegConfig& cfg)
{
    if (X.empty()) throw std::invalid_argument("Empty design matrix.");
    if (X.size() != y.size()) throw std::invalid_argument("X rows and y length must match.");

    const size_t N = X.size();
    const size_t D = X[0].size();

    LogRegModel model;
    model.weights.assign(D, 0.0);
    model.lossHistory.reserve(cfg.epochs);

    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        std::vector<double> grad(D, 0.0);
        double loss = 0.0;

        for (size_t i = 0; i < N; ++i) {
            double p = sigmoid(dot(X[i], model.weights));
            double err = p - y[i];
            for (size_t j = 0; j < D; ++j) {
                grad[j] += err * X[i][j];
            }
            const double eps = 1e-12;
            loss -= y[i] * std::log(p + eps) + (1.0 - y[i]) * std::log(1.0 - p + eps);
        }

        for (size_t j = 0; j < D; ++j) {
            grad[j] /= static_cast<double>(N);
            if (cfg.l2 > 0.0 && j > 0) {
                grad[j] += cfg.l2 * model.weights[j];
            }
            model.weights[j] -= cfg.learningRate * grad[j];
        }
        loss /= static_cast<double>(N);
        model.lossHistory.push_back(loss);

        if (cfg.verbose && (epoch % 100 == 0 || epoch == cfg.epochs - 1)) {
            std::cout << "  epoch " << epoch << "  loss = " << loss << "\n";
        }
    }

    return model;
}

double accuracy(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    const std::vector<double>& weights)
{
    if (X.empty()) return 0.0;
    int correct = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        if (predictClass(X[i], weights) == static_cast<int>(y[i])) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(X.size());
}

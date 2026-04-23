#include "gaussian_discriminant_analysis.h"
#include "stats_and_matrix_operations.h"
#include <cmath>
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

static std::vector<double> matVec(
    const std::vector<std::vector<double>>& M,
    const std::vector<double>& v)
{
    std::vector<double> r(M.size(), 0.0);
    for (size_t i = 0; i < M.size(); ++i) {
        double s = 0.0;
        for (size_t j = 0; j < v.size(); ++j) s += M[i][j] * v[j];
        r[i] = s;
    }
    return r;
}

static double dot(const std::vector<double>& a, const std::vector<double>& b)
{
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

GDAModel fitGDA(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y)
{
    if (X.empty()) throw std::invalid_argument("Empty feature matrix.");
    if (X.size() != y.size()) throw std::invalid_argument("X rows and y length must match.");

    const size_t N = X.size();
    const size_t D = X[0].size();

    GDAModel model;
    model.mu0.assign(D, 0.0);
    model.mu1.assign(D, 0.0);
    model.sigma.assign(D, std::vector<double>(D, 0.0));

    size_t n1 = 0;
    for (size_t i = 0; i < N; ++i) {
        if (y[i] > 0.5) {
            ++n1;
            for (size_t j = 0; j < D; ++j) model.mu1[j] += X[i][j];
        } else {
            for (size_t j = 0; j < D; ++j) model.mu0[j] += X[i][j];
        }
    }
    size_t n0 = N - n1;
    if (n0 == 0 || n1 == 0) throw std::runtime_error("GDA requires both classes present.");

    for (size_t j = 0; j < D; ++j) {
        model.mu0[j] /= static_cast<double>(n0);
        model.mu1[j] /= static_cast<double>(n1);
    }
    model.phi = static_cast<double>(n1) / static_cast<double>(N);

    for (size_t i = 0; i < N; ++i) {
        const auto& mu = (y[i] > 0.5) ? model.mu1 : model.mu0;
        std::vector<double> d(D);
        for (size_t j = 0; j < D; ++j) d[j] = X[i][j] - mu[j];
        for (size_t r = 0; r < D; ++r) {
            for (size_t c = 0; c < D; ++c) {
                model.sigma[r][c] += d[r] * d[c];
            }
        }
    }
    for (size_t r = 0; r < D; ++r) {
        for (size_t c = 0; c < D; ++c) {
            model.sigma[r][c] /= static_cast<double>(N);
        }
        model.sigma[r][r] += 1e-6;
    }

    auto sigmaInv = getMatrixInverse(model.sigma);

    std::vector<double> diff(D);
    for (size_t j = 0; j < D; ++j) diff[j] = model.mu1[j] - model.mu0[j];
    model.w = matVec(sigmaInv, diff);

    auto sInvMu1 = matVec(sigmaInv, model.mu1);
    auto sInvMu0 = matVec(sigmaInv, model.mu0);
    model.b = -0.5 * dot(model.mu1, sInvMu1)
              + 0.5 * dot(model.mu0, sInvMu0)
              + std::log(model.phi / (1.0 - model.phi));

    return model;
}

double predictProbGDA(const std::vector<double>& x, const GDAModel& model)
{
    return sigmoid(dot(model.w, x) + model.b);
}

int predictClassGDA(const std::vector<double>& x, const GDAModel& model, double threshold)
{
    return predictProbGDA(x, model) >= threshold ? 1 : 0;
}

double accuracyGDA(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    const GDAModel& model)
{
    if (X.empty()) return 0.0;
    int correct = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        if (predictClassGDA(X[i], model) == static_cast<int>(y[i])) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(X.size());
}

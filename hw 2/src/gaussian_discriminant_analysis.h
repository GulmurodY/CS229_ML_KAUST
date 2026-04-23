#ifndef GAUSSIAN_DISCRIMINANT_ANALYSIS_H
#define GAUSSIAN_DISCRIMINANT_ANALYSIS_H

#include <vector>

struct GDAModel {
    double phi = 0.0;
    std::vector<double> mu0;
    std::vector<double> mu1;
    std::vector<std::vector<double>> sigma;

    std::vector<double> w;
    double b = 0.0;
};

GDAModel fitGDA(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y);

double predictProbGDA(const std::vector<double>& x, const GDAModel& model);
int    predictClassGDA(const std::vector<double>& x, const GDAModel& model, double threshold = 0.5);

double accuracyGDA(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    const GDAModel& model);

#endif

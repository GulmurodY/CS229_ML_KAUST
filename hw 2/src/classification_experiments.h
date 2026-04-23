#ifndef CLASSIFICATION_EXPERIMENTS_H
#define CLASSIFICATION_EXPERIMENTS_H

#include <string>
#include <vector>

struct StandardizeStats {
    std::vector<double> mean;
    std::vector<double> stddev;
};

StandardizeStats fitStandardizer(const std::vector<std::vector<double>>& X);
std::vector<std::vector<double>> applyStandardizer(
    const std::vector<std::vector<double>>& X,
    const StandardizeStats& s);

std::vector<std::vector<double>> binarizeIris(
    const std::vector<std::vector<double>>& data,
    double positiveClassCode);

void runClassificationExperiment(
    const std::string& datasetName,
    const std::vector<std::vector<double>>& data,
    const std::string& outputDir);

void runSyntheticDecisionBoundary(
    const std::vector<std::vector<double>>& data,
    const std::string& outputDir,
    double gridMin = -5.0,
    double gridMax =  5.0,
    double step    =  0.1);

#endif

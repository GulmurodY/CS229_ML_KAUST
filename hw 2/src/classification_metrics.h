#ifndef CLASSIFICATION_METRICS_H
#define CLASSIFICATION_METRICS_H

#include <vector>
#include <string>

struct ConfusionMatrix {
    int tp = 0;
    int fp = 0;
    int tn = 0;
    int fn = 0;
};

struct ClassificationReport {
    ConfusionMatrix cm;
    double accuracy  = 0.0;
    double precision = 0.0;
    double recall    = 0.0;
    double f1        = 0.0;
};

ConfusionMatrix computeConfusionMatrix(
    const std::vector<int>& yTrue,
    const std::vector<int>& yPred);

ClassificationReport computeReport(
    const std::vector<int>& yTrue,
    const std::vector<int>& yPred);

void printReport(const std::string& name, const ClassificationReport& r);

#endif

#include "classification_metrics.h"
#include <iostream>
#include <stdexcept>

ConfusionMatrix computeConfusionMatrix(
    const std::vector<int>& yTrue,
    const std::vector<int>& yPred)
{
    if (yTrue.size() != yPred.size()) {
        throw std::invalid_argument("yTrue and yPred size mismatch.");
    }
    ConfusionMatrix cm;
    for (size_t i = 0; i < yTrue.size(); ++i) {
        int t = yTrue[i];
        int p = yPred[i];
        if (t == 1 && p == 1) ++cm.tp;
        else if (t == 0 && p == 1) ++cm.fp;
        else if (t == 0 && p == 0) ++cm.tn;
        else if (t == 1 && p == 0) ++cm.fn;
    }
    return cm;
}

ClassificationReport computeReport(
    const std::vector<int>& yTrue,
    const std::vector<int>& yPred)
{
    ClassificationReport r;
    r.cm = computeConfusionMatrix(yTrue, yPred);

    const int total = r.cm.tp + r.cm.fp + r.cm.tn + r.cm.fn;
    r.accuracy = total > 0
        ? static_cast<double>(r.cm.tp + r.cm.tn) / total
        : 0.0;

    const int predPos = r.cm.tp + r.cm.fp;
    const int realPos = r.cm.tp + r.cm.fn;
    r.precision = predPos > 0 ? static_cast<double>(r.cm.tp) / predPos : 0.0;
    r.recall    = realPos > 0 ? static_cast<double>(r.cm.tp) / realPos : 0.0;
    r.f1 = (r.precision + r.recall) > 0.0
        ? 2.0 * r.precision * r.recall / (r.precision + r.recall)
        : 0.0;

    return r;
}

void printReport(const std::string& name, const ClassificationReport& r)
{
    std::cout << "[" << name << "]\n";
    std::cout << "  Confusion matrix  TP=" << r.cm.tp
              << "  FP=" << r.cm.fp
              << "  TN=" << r.cm.tn
              << "  FN=" << r.cm.fn << "\n";
    std::cout << "  Accuracy  = " << r.accuracy  << "\n";
    std::cout << "  Precision = " << r.precision << "\n";
    std::cout << "  Recall    = " << r.recall    << "\n";
    std::cout << "  F1        = " << r.f1        << "\n";
}

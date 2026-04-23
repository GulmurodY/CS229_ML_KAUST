#include "classification_experiments.h"
#include "logistic_regression.h"
#include "gaussian_discriminant_analysis.h"
#include "classification_metrics.h"
#include "stats_and_matrix_operations.h"
#include "linear_regression_models.h"
#include <cmath>
#include <fstream>
#include <iostream>

StandardizeStats fitStandardizer(const std::vector<std::vector<double>>& X)
{
    StandardizeStats s;
    if (X.empty()) return s;
    const size_t N = X.size();
    const size_t D = X[0].size();
    s.mean.assign(D, 0.0);
    s.stddev.assign(D, 0.0);

    for (const auto& row : X)
        for (size_t j = 0; j < D; ++j) s.mean[j] += row[j];
    for (size_t j = 0; j < D; ++j) s.mean[j] /= static_cast<double>(N);

    for (const auto& row : X)
        for (size_t j = 0; j < D; ++j) {
            double d = row[j] - s.mean[j];
            s.stddev[j] += d * d;
        }
    for (size_t j = 0; j < D; ++j) {
        s.stddev[j] = std::sqrt(s.stddev[j] / static_cast<double>(N));
        if (s.stddev[j] < 1e-12) s.stddev[j] = 1.0;
    }
    return s;
}

std::vector<std::vector<double>> applyStandardizer(
    const std::vector<std::vector<double>>& X,
    const StandardizeStats& s)
{
    std::vector<std::vector<double>> out = X;
    for (auto& row : out)
        for (size_t j = 0; j < row.size(); ++j)
            row[j] = (row[j] - s.mean[j]) / s.stddev[j];
    return out;
}

std::vector<std::vector<double>> binarizeIris(
    const std::vector<std::vector<double>>& data,
    double positiveClassCode)
{
    std::vector<std::vector<double>> out;
    out.reserve(data.size());
    for (const auto& row : data) {
        std::vector<double> r = row;
        r.back() = (std::abs(r.back() - positiveClassCode) < 1e-9) ? 1.0 : 0.0;
        out.push_back(std::move(r));
    }
    return out;
}

static void splitFeaturesLabels(
    const std::vector<std::vector<double>>& data,
    std::vector<std::vector<double>>& X,
    std::vector<double>& y)
{
    const size_t N = data.size();
    const size_t D = data[0].size() - 1;
    X.assign(N, std::vector<double>(D));
    y.assign(N, 0.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < D; ++j) {
            X[i][j] = data[i][j];
        }
        y[i] = data[i][D];
    }
}

static std::vector<std::vector<double>> addBiasColumn(
    const std::vector<std::vector<double>>& X)
{
    std::vector<std::vector<double>> out(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        out[i].reserve(X[i].size() + 1);
        out[i].push_back(1.0);
        for (double v : X[i]) { 
            out[i].push_back(v);
        }
    }
    return out;
}

static void writeLossCurveCSV(
    const std::string& path,
    const std::vector<double>& losses)
{
    std::ofstream f(path);
    f << "epoch,loss\n";
    for (size_t i = 0; i < losses.size(); ++i)
        f << i << "," << losses[i] << "\n";
}

static void writePredictionsCSV(
    const std::string& path,
    const std::vector<int>& yTrue,
    const std::vector<int>& yPred,
    const std::vector<double>& prob)
{
    std::ofstream f(path);
    f << "y_true,y_pred,prob\n";
    for (size_t i = 0; i < yTrue.size(); ++i)
        f << yTrue[i] << "," << yPred[i] << "," << prob[i] << "\n";
}

static void writeConfusionCSV(
    const std::string& path,
    const ClassificationReport& r)
{
    std::ofstream f(path);
    f << "TP,FP,TN,FN,accuracy,precision,recall,f1\n";
    f << r.cm.tp << "," << r.cm.fp << "," << r.cm.tn << "," << r.cm.fn << ","
      << r.accuracy << "," << r.precision << "," << r.recall << "," << r.f1 << "\n";
}

void runClassificationExperiment(
    const std::string& datasetName,
    const std::vector<std::vector<double>>& data,
    const std::string& outputDir)
{
    std::cout << "\n=== Classification experiment: " << datasetName << " ===\n";

    auto split = splitData(data, 0.8);

    std::vector<std::vector<double>> XtrRaw, XvaRaw;
    std::vector<double> ytr, yva;
    splitFeaturesLabels(split.train,      XtrRaw, ytr);
    splitFeaturesLabels(split.validation, XvaRaw, yva);

    auto std = fitStandardizer(XtrRaw);
    auto XtrZ = applyStandardizer(XtrRaw, std);
    auto XvaZ = applyStandardizer(XvaRaw, std);

    auto XtrLR = addBiasColumn(XtrZ);
    auto XvaLR = addBiasColumn(XvaZ);

    LogRegConfig cfg;
    cfg.learningRate = 0.1;
    cfg.epochs       = 1000;
    auto lrModel = fitLogisticRegression(XtrLR, ytr, cfg);

    GDAModel gdaModel;
    bool gdaOk = true;
    try {
        gdaModel = fitGDA(XtrZ, ytr);
    } catch (const std::exception& e) {
        std::cout << "  GDA fit failed: " << e.what() << "\n";
        gdaOk = false;
    }

    std::vector<int> yTrue(yva.size());
    std::vector<int> yPredLR(yva.size());
    std::vector<double> probLR(yva.size());
    for (size_t i = 0; i < yva.size(); ++i) {
        yTrue[i]  = static_cast<int>(yva[i]);
        probLR[i] = predictProb(XvaLR[i], lrModel.weights);
        yPredLR[i] = probLR[i] >= 0.5 ? 1 : 0;
    }
    auto reportLR = computeReport(yTrue, yPredLR);
    printReport(datasetName + " / LogReg", reportLR);

    writeLossCurveCSV (outputDir + "/classification_loss_"        + datasetName + ".csv", lrModel.lossHistory);
    writePredictionsCSV(outputDir + "/classification_predictions_" + datasetName + "_logreg.csv", yTrue, yPredLR, probLR);
    writeConfusionCSV (outputDir + "/classification_confusion_"   + datasetName + "_logreg.csv", reportLR);

    ClassificationReport reportGDA;
    if (gdaOk) {
        std::vector<int> yPredGDA(yva.size());
        std::vector<double> probGDA(yva.size());
        for (size_t i = 0; i < yva.size(); ++i) {
            probGDA[i]  = predictProbGDA(XvaZ[i], gdaModel);
            yPredGDA[i] = probGDA[i] >= 0.5 ? 1 : 0;
        }
        reportGDA = computeReport(yTrue, yPredGDA);
        printReport(datasetName + " / GDA   ", reportGDA);

        writePredictionsCSV(outputDir + "/classification_predictions_" + datasetName + "_gda.csv", yTrue, yPredGDA, probGDA);
        writeConfusionCSV (outputDir + "/classification_confusion_"   + datasetName + "_gda.csv",  reportGDA);
    }

    std::ofstream summary(outputDir + "/classification_summary.csv", std::ios::app);
    summary << datasetName << ",logreg," << reportLR.accuracy << "," << reportLR.precision
            << "," << reportLR.recall << "," << reportLR.f1 << "\n";
    if (gdaOk) {
        summary << datasetName << ",gda," << reportGDA.accuracy << "," << reportGDA.precision
                << "," << reportGDA.recall << "," << reportGDA.f1 << "\n";
    }
}

void runSyntheticDecisionBoundary(
    const std::vector<std::vector<double>>& data,
    const std::string& outputDir,
    double gridMin,
    double gridMax,
    double step)
{
    std::cout << "\n=== Decision boundary grid (synthetic 2D) ===\n";

    std::vector<std::vector<double>> X;
    std::vector<double> y;
    splitFeaturesLabels(data, X, y);

    auto Xlr = addBiasColumn(X);
    LogRegConfig cfg;
    cfg.learningRate = 0.1;
    cfg.epochs       = 1000;
    auto lrModel  = fitLogisticRegression(Xlr, y, cfg);
    auto gdaModel = fitGDA(X, y);

    {
        std::ofstream f(outputDir + "/classification_synthetic_data.csv");
        f << "x1,x2,y\n";
        for (const auto& row : data) f << row[0] << "," << row[1] << "," << row[2] << "\n";
    }

    std::ofstream glr(outputDir + "/classification_synthetic_grid_logreg.csv");
    std::ofstream gga(outputDir + "/classification_synthetic_grid_gda.csv");
    glr << "x1,x2,prob\n";
    gga << "x1,x2,prob\n";

    size_t pts = 0;
    for (double x1 = gridMin; x1 <= gridMax + 1e-12; x1 += step) {
        for (double x2 = gridMin; x2 <= gridMax + 1e-12; x2 += step) {
            std::vector<double> xLR = {1.0, x1, x2};
            std::vector<double> xG  = {x1, x2};
            double pLR = predictProb(xLR, lrModel.weights);
            double pG  = predictProbGDA(xG, gdaModel);
            glr << x1 << "," << x2 << "," << pLR << "\n";
            gga << x1 << "," << x2 << "," << pG  << "\n";
            ++pts;
        }
    }
    std::cout << "  Wrote " << pts << " grid points per model\n";
    std::cout << "  LogReg weights: bias=" << lrModel.weights[0]
              << "  w1=" << lrModel.weights[1]
              << "  w2=" << lrModel.weights[2] << "\n";
    std::cout << "  GDA    weights: b="    << gdaModel.b
              << "  w1=" << gdaModel.w[0]
              << "  w2=" << gdaModel.w[1] << "\n";
}

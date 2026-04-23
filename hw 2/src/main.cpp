#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include "stats_and_matrix_operations.h"
#include "linear_regression_models.h"
#include "basis_functions.h"
#include "csv_data_loader.h"
#include "logistic_regression.h"
#include "gaussian_discriminant_analysis.h"
#include "classification_metrics.h"
#include "classification_experiments.h"
#include <fstream>

static void extractFeaturesAndLabels(
    const std::vector<std::vector<double>>& data,
    std::vector<std::vector<double>>& X,
    std::vector<double>& y)
{
    const size_t N = data.size();
    const size_t D = data[0].size() - 1;
    X.assign(N, std::vector<double>(D));
    y.assign(N, 0.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < D; ++j) X[i][j] = data[i][j];
        y[i] = data[i][D];
    }
}

static std::vector<std::vector<double>> generateLinearlySeparable2D(
    int nPerClass = 100, double spread = 0.8, unsigned seed = 7)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, spread);
    std::vector<std::vector<double>> data;
    data.reserve(2 * nPerClass);
    for (int i = 0; i < nPerClass; ++i) {
        data.push_back({-2.0 + noise(rng), -2.0 + noise(rng), 0.0});
    }
    for (int i = 0; i < nPerClass; ++i) {
        data.push_back({ 2.0 + noise(rng),  2.0 + noise(rng), 1.0});
    }
    std::shuffle(data.begin(), data.end(), rng);
    return data;
}

std::vector<std::vector<double>> generateSyntheticData(
    int N = 50, 
    double noise_std = 0.2, 
    unsigned seed = 42) 
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> x_dist(0.0, 1.0);
    std::normal_distribution<double> noise_dist(0.0, noise_std);

    const double PI = std::acos(-1.0);
    std::vector<std::vector<double>> data;
    data.reserve(N);

    for (int i = 0; i < N; ++i) {
        double x = x_dist(rng);
        double y = std::sin(2.0 * PI * x) + noise_dist(rng);
        data.push_back({x, y});
    }
    return data;
}


int main()
{
    // Classification Step 1: Verify Iris + Titanic loaders with printSummary.
    {
        std::cout << "=== Classification Data Loading ===\n";

        checkDatasetLoading("Titanic", titanicPath, getConfigForDataset(titanicPath), 8);
        checkDatasetLoading("Iris",    irisPath,    getConfigForDataset(irisPath),    5);
    }

    // Classification Step 2: Verify logistic regression on a linearly-separable synthetic set.
    {
        std::cout << "\n=== LogReg sanity check (linearly separable 2D) ===\n";

        auto data  = generateLinearlySeparable2D(100, 0.8, 7);
        auto split = splitData(data, 0.8);

        std::vector<std::vector<double>> Xtr, Xva;
        std::vector<double> ytr, yva;
        getModelMatrices(split.train,      Xtr, ytr);
        getModelMatrices(split.validation, Xva, yva);

        LogRegConfig cfg;
        cfg.learningRate = 0.1;
        cfg.epochs       = 500;
        cfg.verbose      = true;

        auto model = fitLogisticRegression(Xtr, ytr, cfg);

        std::cout << "Weights (bias, w1, w2): "
                  << model.weights[0] << ", "
                  << model.weights[1] << ", "
                  << model.weights[2] << "\n";
        std::cout << "Train accuracy = " << accuracy(Xtr, ytr, model.weights) << "\n";
        std::cout << "Val   accuracy = " << accuracy(Xva, yva, model.weights) << "\n";
    }

    // Classification Step 3: Verify GDA on the same linearly-separable synthetic set.
    {
        std::cout << "\n=== GDA sanity check (linearly separable 2D) ===\n";

        auto data  = generateLinearlySeparable2D(100, 0.8, 7);
        auto split = splitData(data, 0.8);

        std::vector<std::vector<double>> Xtr, Xva;
        std::vector<double> ytr, yva;
        extractFeaturesAndLabels(split.train,      Xtr, ytr);
        extractFeaturesAndLabels(split.validation, Xva, yva);

        auto model = fitGDA(Xtr, ytr);

        std::cout << "phi   = " << model.phi << "\n";
        std::cout << "mu0   = (" << model.mu0[0] << ", " << model.mu0[1] << ")\n";
        std::cout << "mu1   = (" << model.mu1[0] << ", " << model.mu1[1] << ")\n";
        std::cout << "w     = (" << model.w[0]   << ", " << model.w[1]   << ")\n";
        std::cout << "b     = " << model.b << "\n";
        std::cout << "Train accuracy = " << accuracyGDA(Xtr, ytr, model) << "\n";
        std::cout << "Val   accuracy = " << accuracyGDA(Xva, yva, model) << "\n";
    }

    // Classification Step 4: Metrics + confusion matrix on both models.
    {
        std::cout << "\n=== Metrics sanity check (LogReg + GDA on separable 2D) ===\n";

        auto data  = generateLinearlySeparable2D(100, 0.8, 7);
        auto split = splitData(data, 0.8);

        std::vector<std::vector<double>> Xtr_lr, Xva_lr;
        std::vector<double> ytr_lr, yva_lr;
        getModelMatrices(split.train,      Xtr_lr, ytr_lr);
        getModelMatrices(split.validation, Xva_lr, yva_lr);

        std::vector<std::vector<double>> Xtr_g, Xva_g;
        std::vector<double> ytr_g, yva_g;
        extractFeaturesAndLabels(split.train,      Xtr_g, ytr_g);
        extractFeaturesAndLabels(split.validation, Xva_g, yva_g);

        LogRegConfig cfg;
        cfg.learningRate = 0.1;
        cfg.epochs       = 500;
        auto lrModel  = fitLogisticRegression(Xtr_lr, ytr_lr, cfg);
        auto gdaModel = fitGDA(Xtr_g, ytr_g);

        std::vector<int> yTrue(yva_lr.size());
        std::vector<int> yPredLR(yva_lr.size());
        std::vector<int> yPredGDA(yva_lr.size());
        for (size_t i = 0; i < yva_lr.size(); ++i) {
            yTrue[i]    = static_cast<int>(yva_lr[i]);
            yPredLR[i]  = predictClass(Xva_lr[i], lrModel.weights);
            yPredGDA[i] = predictClassGDA(Xva_g[i], gdaModel);
        }

        printReport("LogReg (val)", computeReport(yTrue, yPredLR));
        printReport("GDA    (val)", computeReport(yTrue, yPredGDA));
    }

    // Classification Step 5: Experiment loop on Titanic and Iris, writing CSVs.
    {
        const std::string outDir = "output_csv";
        {
            std::ofstream f(outDir + "/classification_summary.csv");
            f << "dataset,model,accuracy,precision,recall,f1\n";
        }

        auto titanic = loadCSV(titanicPath, getConfigForDataset(titanicPath));
        runClassificationExperiment("titanic", titanic, outDir);

        auto iris = loadCSV(irisPath, getConfigForDataset(irisPath));
        auto irisBinary = binarizeIris(iris, 1.0);
        runClassificationExperiment("iris_setosa_vs_rest", irisBinary, outDir);

        auto noisyBlobs = generateLinearlySeparable2D(200, 1.5, 11);
        runSyntheticDecisionBoundary(noisyBlobs, outDir);
    }

    // Experiment #1: Fit Bayesian LR with polynomial basis on synthetic sin data and evaluate training vs validation error.
    {
        std::cout << "=== Synthetic 1D Dataset (y = sin(2*pi*x) + noise) ===\n";

        auto data  = generateSyntheticData(50, 0.2, 42);
        auto split = splitData(data, 0.8);

        BasisTransformConfig basisCfg;
        basisCfg.basisType = "polynomial";
        basisCfg.polynomialDegree = 9;

        auto model = computePosterior(split.train, 1e-3, 25.0, basisCfg);

        double trainingMSE   = bayesianMSE(split.train, model);
        double validationMSE = bayesianMSE(split.validation, model);

        std::cout << "Polynomial basis (degree 9):\n";
        std::cout << "  Training MSE   = " << trainingMSE   << "\n";
        std::cout << "  Validation MSE = " << validationMSE << "\n";

        savePredictionsToCSV("output_csv/synthetic_predictions.csv", data, model);
        savePosteriorToCSV("output_csv/synthetic_posterior.csv", model);
    }

    // Experiment #2: Compare polynomial, Gaussian, and sigmoidal basis functions on the Wine dataset.
    {
        std::cout << "=== Wine Dataset — Basis Function Comparison ===\n";

        auto data  = loadCSV(winePath, getConfigForDataset(winePath));
        auto split = splitData(data, 0.8);

        std::vector<std::pair<std::string, BasisTransformConfig>> basisOptions;

        BasisTransformConfig polyCfg;
        polyCfg.basisType = "polynomial";
        polyCfg.polynomialDegree = 2;
        basisOptions.push_back({"polynomial", polyCfg});

        BasisTransformConfig gaussCfg;
        gaussCfg.basisType = "gaussian";
        gaussCfg.center = 0.0;
        gaussCfg.scale  = 1.0;
        basisOptions.push_back({"gaussian", gaussCfg});

        BasisTransformConfig sigmoidCfg;
        sigmoidCfg.basisType = "sigmoidal";
        sigmoidCfg.center = 0.0;
        sigmoidCfg.scale  = 1.0;
        basisOptions.push_back({"sigmoidal", sigmoidCfg});

        std::vector<std::string> mseLabels;
        std::vector<double> mseValues;

        for (auto& [name, basisCfg] : basisOptions) {
            auto model = computePosterior(split.train, 1e-3, 1.0, basisCfg);
            double trainingMSE   = bayesianMSE(split.train, model);
            double validationMSE = bayesianMSE(split.validation, model);

            std::cout << name << " basis:\n";
            std::cout << "  Training MSE   = " << trainingMSE   << "\n";
            std::cout << "  Validation MSE = " << validationMSE << "\n";

            savePredictionsToCSV("output_csv/wine_" + name + "_predictions.csv", data, model);
            savePosteriorToCSV("output_csv/wine_" + name + "_posterior.csv", model);

            mseLabels.push_back("wine_" + name + "_training_MSE");
            mseValues.push_back(trainingMSE);
            mseLabels.push_back("wine_" + name + "_validation_MSE");
            mseValues.push_back(validationMSE);
        }

        saveResultsToCSV("output_csv/wine_mse_comparison.csv", mseLabels, mseValues);
    }

    // Experiment #3: Apply Bayesian LR with polynomial basis to the Automobile dataset to predict car prices.
    {
        std::cout << "=== Automobile Dataset ===\n";

        auto data  = loadCSV(automobilePath, getConfigForDataset(automobilePath));
        auto split = splitData(data, 0.8);

        BasisTransformConfig basisCfg;
        basisCfg.basisType = "polynomial";
        basisCfg.polynomialDegree = 2;

        auto model = computePosterior(split.train, 1e-3, 1e-6, basisCfg);
        double trainingMSE   = bayesianMSE(split.train, model);
        double validationMSE = bayesianMSE(split.validation, model);

        std::cout << "Polynomial basis (degree 2):\n";
        std::cout << "  Training MSE   = " << trainingMSE   << "\n";
        std::cout << "  Validation MSE = " << validationMSE << "\n";

        savePredictionsToCSV("output_csv/automobile_predictions.csv", data, model);
        savePosteriorToCSV("output_csv/automobile_posterior.csv", model);
    }

    // Experiment #4: Compare MLE closed-form solution against Bayesian LR on Wine and Automobile datasets.
    {
        std::cout << "=== MLE vs Bayesian Comparison ===\n";

        {
            auto data  = loadCSV(winePath, getConfigForDataset(winePath));
            auto split = splitData(data, 0.8);

            auto mleWeights        = linearRegressionClosedForm(applyPolynomialExpansion(split.train, 2), 1e-3);
            double mleTrainingMSE   = calculateMSE(applyPolynomialExpansion(split.train, 2), mleWeights);
            double mleValidationMSE = calculateMSE(applyPolynomialExpansion(split.validation, 2), mleWeights);

            BasisTransformConfig basisCfg;
            basisCfg.basisType = "polynomial";
            basisCfg.polynomialDegree = 2;
            auto model = computePosterior(split.train, 1e-3, 1.0, basisCfg);
            double bayesTrainingMSE   = bayesianMSE(split.train, model);
            double bayesValidationMSE = bayesianMSE(split.validation, model);

            std::cout << "[Wine, polynomial degree 2]\n";
            std::cout << "  MLE      Training MSE = " << mleTrainingMSE   << "  Validation MSE = " << mleValidationMSE   << "\n";
            std::cout << "  Bayesian Training MSE = " << bayesTrainingMSE << "  Validation MSE = " << bayesValidationMSE << "\n\n";

            saveResultsToCSV("output_csv/wine_mle_vs_bayes.csv",
                {"MLE_training", "MLE_validation", "Bayesian_training", "Bayesian_validation"},
                {mleTrainingMSE, mleValidationMSE, bayesTrainingMSE, bayesValidationMSE});
        }

        {
            auto data  = loadCSV(automobilePath, getConfigForDataset(automobilePath));
            auto split = splitData(data, 0.8);

            auto mleWeights        = linearRegressionClosedForm(applyPolynomialExpansion(split.train, 2), 1e-3);
            double mleTrainingMSE   = calculateMSE(applyPolynomialExpansion(split.train, 2), mleWeights);
            double mleValidationMSE = calculateMSE(applyPolynomialExpansion(split.validation, 2), mleWeights);

            BasisTransformConfig basisCfg;
            basisCfg.basisType = "polynomial";
            basisCfg.polynomialDegree = 2;
            auto model = computePosterior(split.train, 1e-3, 1e-6, basisCfg);
            double bayesTrainingMSE   = bayesianMSE(split.train, model);
            double bayesValidationMSE = bayesianMSE(split.validation, model);

            std::cout << "[Automobile, polynomial degree 2]\n";
            std::cout << "  MLE      Training MSE = " << mleTrainingMSE   << "  Validation MSE = " << mleValidationMSE   << "\n";
            std::cout << "  Bayesian Training MSE = " << bayesTrainingMSE << "  Validation MSE = " << bayesValidationMSE << "\n\n";

            saveResultsToCSV("output_csv/automobile_mle_vs_bayes.csv",
                {"MLE_training", "MLE_validation", "Bayesian_training", "Bayesian_validation"},
                {mleTrainingMSE, mleValidationMSE, bayesTrainingMSE, bayesValidationMSE});
        }

    }

    return 0;
}

#include <iostream>
#include <vector>
#include <random>
#include <string>
#include "stats_and_matrix_operations.h"
#include "linear_regression_models.h"
#include "basis_functions.h"
#include "csv_data_loader.h"

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

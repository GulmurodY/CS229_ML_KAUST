#include <iostream>
#include <vector>
#include <sstream>
#include <random>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include <stdexcept>
#include "stats_and_matrix_operations.h"
#include "linear_regression_models.h"
#include "basis_functions.h"


std::vector<std::vector<double>> generateSyntheticData(int N = 50, double noise_std = 0.2, unsigned seed = 42) {
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

const std::string titanicPath = "data_for_classification/titanic/train.csv";
const std::string irisPath = "data_for_classification/Iris/Iris.csv";
const std::string winePath = "data_for_regression/wine/winequality-red.csv";
const std::string forestFiresPath = "data_for_regression/forest fires/forestfires.csv";
const std::string automobilePath = "data_for_regression/automobile/imports-85.data";


int monthToNumber(const std::string& month) {
    std::map<std::string, int> months = {
        {"jan", 1}, {"feb", 2}, {"mar", 3}, {"apr", 4},
        {"may", 5}, {"jun", 6}, {"jul", 7}, {"aug", 8},
        {"sep", 9}, {"oct", 10}, {"nov", 11}, {"dec", 12}
    };
    auto it = months.find(month);
    return (it != months.end()) ? it->second : 1;
}

int dayToNumber(const std::string& day) {
    std::map<std::string, int> days = {
        {"mon", 1}, {"tue", 2}, {"wed", 3}, {"thu", 4},
        {"fri", 5}, {"sat", 6}, {"sun", 7}
    };
    auto it = days.find(day);
    return (it != days.end()) ? it->second : 1;
}

struct CSVConfig {
    char delimiter = ',';
    bool hasHeader = true;

    std::set<int> skipColumns;

    std::map<int, std::map<std::string, double>> categoricalMaps;
    
    int targetColumn = -1;


    double missingDefault = 0.0;
};

static std::vector<std::string> parseCSVLine(const std::string& line, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    bool inQuotes = false;
    for (char c : line) {
        if (c == '"') {
            inQuotes = !inQuotes;
        } else if (c == delimiter && !inQuotes) {
            tokens.push_back(token);
            token.clear();
        } else {
            token += c;
        }
    }
    tokens.push_back(token);
    return tokens;
}

std::vector<std::vector<double>> loadCSV(const std::string& filename, const CSVConfig& cfg) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return data;
    }

    std::string line;
    if (cfg.hasHeader) {
        std::getline(file, line);
    }

    while (std::getline(file, line)) {
        auto tokens = parseCSVLine(line, cfg.delimiter);

        std::vector<double> features;
        double target        =  cfg.missingDefault    ;
        bool   hasTarget     = (cfg.targetColumn >= 0);

        for (int col = 0; col < static_cast<int>(tokens.size()); ++col) {
            if (cfg.skipColumns.count(col)) continue;

            const std::string& token = tokens[col];
            double value = cfg.missingDefault;

            if (!token.empty()) {
                auto catIt = cfg.categoricalMaps.find(col);
                if (catIt != cfg.categoricalMaps.end()) {
                    auto encIt = catIt->second.find(token);
                    value = (encIt != catIt->second.end()) ? encIt->second : cfg.missingDefault;
                } else {
                    try { value = std::stod(token); } catch (...) { value = cfg.missingDefault; }
                }
            }

            if (hasTarget && col == cfg.targetColumn) {
                target = value;
            } else {
                features.push_back(value);
            }
        }

        if (features.empty()) continue;
        if (hasTarget) features.push_back(target);
        data.push_back(std::move(features));
    }

    return data;
}

bool printSummary(const std::string& name,
                  const std::vector<std::vector<double>>& data,
                  std::size_t expectedColumns = 0) {
    std::cout << "\n[" << name << "]\n";
    std::cout << "Rows: " << data.size() << "\n";
    std::cout << "Columns (features + target): "
                << (data.empty() ? 0 : data[0].size()) << "\n";

    if (data.empty()) {
        std::cout << "Status: FAILED (no rows loaded)\n";
        return false;
    }

    std::cout << "First parsed row: ";
    for (double v : data.front()) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    std::set<double> uniqueTargets;
    for (const auto& row : data) {
        uniqueTargets.insert(row.back());
    }

    std::cout << "Unique target values count: " << uniqueTargets.size() << "\n";

    bool passed = true;
    if (expectedColumns > 0 && data[0].size() != expectedColumns) {
        passed = false;
        std::cout << "Expected columns: " << expectedColumns << "\n";
        std::cout << "Status: FAILED (column count mismatch)\n";
    } else {
        std::cout << "Status: PASSED (dataset loaded and target moved to last column)\n";
    }

    return passed;
}

void printStatisticalFeatures(const std::string& name, const std::vector<std::vector<double>>& data) {
    if (data.empty() || data[0].empty()) {
        return;
    }

    const std::size_t columnCount = data[0].size();
    const std::size_t targetIndex = columnCount - 1;

    std::cout << "Statistical features for " << name << " (features only):\n";
    for (std::size_t col = 0; col < targetIndex; ++col) {
        std::vector<double> values;
        values.reserve(data.size());
        for (const auto& row : data) {
            values.push_back(row[col]);
        }

        double mean = getMean(values);
        double variance = getVariance(values);
        double stdDev = getStandardDeviation(values);

        std::cout << "  Feature[" << col << "]"
                  << " mean=" << mean
                  << ", variance=" << variance
                  << ", stdDev=" << stdDev << "\n";
    }
}

bool checkDatasetLoading(const std::string& name, const std::string& path, const CSVConfig& cfg, std::size_t expectedColumns) {
    if (path.empty()) {
        std::cout << "\n[" << name << "]\n";
        std::cout << "Status: FAILED (dataset path not found)\n";
        return false;
    }

    auto data = loadCSV(path, cfg);
    bool passed = printSummary(name, data, expectedColumns);
    if (passed) {
        printStatisticalFeatures(name, data);
    }
    return passed;
}

bool runBasisFunctionSanityTest() {
    const double tol = 1e-9;
    bool passed = true;

    const double gaussianAtMean = gaussianBasis(0.0, 0.0, 1.0);
    const double expectedGaussianAtMean = 1.0;
    if (std::abs(gaussianAtMean - expectedGaussianAtMean) > tol) {
        passed = false;
    }

    const double sigmoidAtCenter = sigmoidalBasis(0.0, 0.0, 1.0);
    const double expectedSigmoidAtCenter = 0.5;
    if (std::abs(sigmoidAtCenter - expectedSigmoidAtCenter) > tol) {
        passed = false;
    }

    std::vector<std::vector<double>> tinyData = {
        {0.0, 10.0},
        {1.0, 20.0}
    };

    auto gaussianData = applyBasisFunctions(tinyData, "gaussian", 0.0, 1.0);
    auto sigmoidalData = applyBasisFunctions(tinyData, "sigmoidal", 0.0, 1.0);

    const double expectedGaussianSecond = std::exp(-0.5);
    if (std::abs(gaussianData[0][0] - 1.0) > tol ||
        std::abs(gaussianData[1][0] - expectedGaussianSecond) > tol) {
        passed = false;
    }

    if (std::abs(sigmoidalData[0][0] - 0.5) > tol) {
        passed = false;
    }

    if (gaussianData[0][1] != 10.0 || gaussianData[1][1] != 20.0 ||
        sigmoidalData[0][1] != 10.0 || sigmoidalData[1][1] != 20.0) {
        passed = false;
    }

    std::cout << "\n[Basis Functions Sanity Test]\n";
    std::cout << (passed ? "Status: PASSED\n" : "Status: FAILED\n");
    return passed;
}

CSVConfig getConfigForDataset(const std::string& filePath) {
    static const std::map<std::string, CSVConfig> configMap = []() {
        std::map<std::string, CSVConfig> m;

        CSVConfig titanicCfg;
        titanicCfg.skipColumns  = {0, 3, 8, 10};
        titanicCfg.targetColumn = 1;
        titanicCfg.categoricalMaps = {
            {4,  {{"male", 0.0}, {"female", 1.0}}},
            {11, {{"S", 1.0}, {"C", 2.0}, {"Q", 3.0}}}
        };
        m[titanicPath] = titanicCfg;

        CSVConfig irisCfg;
        irisCfg.skipColumns  = {0};
        irisCfg.targetColumn = 5;
        irisCfg.categoricalMaps = {
            {5, {{"Iris-setosa", 1.0}, {"Iris-versicolor", 2.0}, {"Iris-virginica", 3.0}}}
        };
        m[irisPath] = irisCfg;

        CSVConfig wineCfg;
        wineCfg.delimiter = ';';
        wineCfg.targetColumn = -1;
        m[winePath] = wineCfg;

        CSVConfig forestCfg;
        forestCfg.targetColumn = 12;
        forestCfg.categoricalMaps = {
            {2, {{"jan", 1.0}, {"feb", 2.0}, {"mar", 3.0}, {"apr", 4.0},
                 {"may", 5.0}, {"jun", 6.0}, {"jul", 7.0}, {"aug", 8.0},
                 {"sep", 9.0}, {"oct", 10.0}, {"nov", 11.0}, {"dec", 12.0}}},
            {3, {{"mon", 1.0}, {"tue", 2.0}, {"wed", 3.0}, {"thu", 4.0},
                 {"fri", 5.0}, {"sat", 6.0}, {"sun", 7.0}}}
        };
        m[forestFiresPath] = forestCfg;

        CSVConfig automobileCfg;
        automobileCfg.hasHeader = false;
        automobileCfg.targetColumn = 25;
        automobileCfg.skipColumns = {2, 3, 4, 5, 6, 7, 8, 14, 15, 17};
        m[automobilePath] = automobileCfg;

        return m;
    }();

    auto it = configMap.find(filePath);
    if (it == configMap.end()) {
        throw std::invalid_argument("No CSV config defined for path: " + filePath);
    }
    return it->second;
}

double bayesianMSE(const std::vector<std::vector<double>>& data,
                   const BayesianLinearRegression& model)
{
    auto Phi = buildDesignMatrix(data, model.basisConfig);
    double total = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        double err = data[i].back() - predictiveMean(Phi[i], model.m_N);
        total += err * err;
    }
    return total / data.size();
}

int main() {
    std::cout << "=== Synthetic 1D Dataset (y = sin(2*pi*x) + noise) ===\n";

    auto synData = generateSyntheticData(50, 0.2, 42);
    auto synSplit = splitData(synData, 0.8);

    BasisTransformConfig polyCfg;
    polyCfg.basisType = "polynomial";
    polyCfg.polynomialDegree = 9;

    auto synModel = computePosterior(synSplit.train, 1e-3, 25.0, polyCfg);

    double synTrainMSE = bayesianMSE(synSplit.train, synModel);
    double synValMSE   = bayesianMSE(synSplit.validation, synModel);

    std::cout << "Polynomial basis (degree 9):\n";
    std::cout << "  Train MSE = " << synTrainMSE << "\n";
    std::cout << "  Val   MSE = " << synValMSE   << "\n";

    savePredictionsToCSV("output_csv/synthetic_predictions.csv", synData, synModel);
    savePosteriorToCSV("output_csv/synthetic_posterior.csv", synModel);

    std::cout << "Saved: output_csv/synthetic_predictions.csv\n";
    std::cout << "Saved: output_csv/synthetic_posterior.csv\n\n";

    std::cout << "=== Wine Dataset — Basis Function Comparison ===\n";

    CSVConfig wineCfg = getConfigForDataset(winePath);
    auto wineData = loadCSV(winePath, wineCfg);
    auto wineSplit = splitData(wineData, 0.8);

    std::vector<std::pair<std::string, BasisTransformConfig>> basisOptions;

    BasisTransformConfig winePolyCfg;
    winePolyCfg.basisType = "polynomial";
    winePolyCfg.polynomialDegree = 2;
    basisOptions.push_back({"polynomial", winePolyCfg});

    BasisTransformConfig wineGaussCfg;
    wineGaussCfg.basisType = "gaussian";
    wineGaussCfg.center = 0.0;
    wineGaussCfg.scale  = 1.0;
    basisOptions.push_back({"gaussian", wineGaussCfg});

    BasisTransformConfig wineSigCfg;
    wineSigCfg.basisType = "sigmoidal";
    wineSigCfg.center = 0.0;
    wineSigCfg.scale  = 1.0;
    basisOptions.push_back({"sigmoidal", wineSigCfg});

    std::vector<std::string> mseLabels;
    std::vector<double> mseValues;

    for (auto& [name, cfg] : basisOptions) {
        auto model = computePosterior(wineSplit.train, 1e-3, 1.0, cfg);
        double trainMSE = bayesianMSE(wineSplit.train, model);
        double valMSE   = bayesianMSE(wineSplit.validation, model);

        std::cout << name << " basis:\n";
        std::cout << "  Train MSE = " << trainMSE << "\n";
        std::cout << "  Val   MSE = " << valMSE   << "\n";

        savePredictionsToCSV("output_csv/wine_" + name + "_predictions.csv", wineData, model);
        savePosteriorToCSV("output_csv/wine_" + name + "_posterior.csv", model);

        mseLabels.push_back("wine_" + name + "_train_MSE");
        mseValues.push_back(trainMSE);
        mseLabels.push_back("wine_" + name + "_val_MSE");
        mseValues.push_back(valMSE);
    }

    saveResultsToCSV("output_csv/wine_mse_comparison.csv", mseLabels, mseValues);
    std::cout << "Saved: output_csv/wine_mse_comparison.csv\n\n";

    std::cout << "=== Automobile Dataset ===\n";

    CSVConfig autoCfg = getConfigForDataset(automobilePath);
    auto autoData = loadCSV(automobilePath, autoCfg);
    auto autoSplit = splitData(autoData, 0.8);

    BasisTransformConfig autoPolyCfg;
    autoPolyCfg.basisType = "polynomial";
    autoPolyCfg.polynomialDegree = 2;

    auto autoModel = computePosterior(autoSplit.train, 1e-3, 1e-6, autoPolyCfg);
    double autoTrainMSE = bayesianMSE(autoSplit.train, autoModel);
    double autoValMSE   = bayesianMSE(autoSplit.validation, autoModel);

    std::cout << "Polynomial basis (degree 2):\n";
    std::cout << "  Train MSE = " << autoTrainMSE << "\n";
    std::cout << "  Val   MSE = " << autoValMSE   << "\n";

    savePredictionsToCSV("output_csv/automobile_predictions.csv", autoData, autoModel);
    savePosteriorToCSV("output_csv/automobile_posterior.csv", autoModel);
    std::cout << "Saved: output_csv/automobile_predictions.csv\n\n";

    std::cout << "=== MLE vs Bayesian Comparison ===\n";

    {
        auto mleWeights = linearRegressionClosedForm(
            applyPolynomialExpansion(wineSplit.train, 2), 1e-3);
        double mleTrain = calculateMSE(applyPolynomialExpansion(wineSplit.train, 2), mleWeights);
        double mleVal   = calculateMSE(applyPolynomialExpansion(wineSplit.validation, 2), mleWeights);

        BasisTransformConfig cfg; cfg.basisType = "polynomial"; cfg.polynomialDegree = 2;
        auto bayModel = computePosterior(wineSplit.train, 1e-3, 1.0, cfg);
        double bayTrain = bayesianMSE(wineSplit.train, bayModel);
        double bayVal   = bayesianMSE(wineSplit.validation, bayModel);

        std::cout << "[Wine, polynomial degree 2]\n";
        std::cout << "  MLE      Train MSE = " << mleTrain << "  Val MSE = " << mleVal << "\n";
        std::cout << "  Bayesian Train MSE = " << bayTrain << "  Val MSE = " << bayVal << "\n\n";

        saveResultsToCSV("output_csv/wine_mle_vs_bayes.csv",
            {"MLE_train", "MLE_val", "Bayes_train", "Bayes_val"},
            {mleTrain, mleVal, bayTrain, bayVal});
    }

    {
        auto mleWeights = linearRegressionClosedForm(
            applyPolynomialExpansion(autoSplit.train, 2), 1e-3);
        double mleTrain = calculateMSE(applyPolynomialExpansion(autoSplit.train, 2), mleWeights);
        double mleVal   = calculateMSE(applyPolynomialExpansion(autoSplit.validation, 2), mleWeights);

        std::cout << "[Automobile, polynomial degree 2]\n";
        std::cout << "  MLE      Train MSE = " << mleTrain << "  Val MSE = " << mleVal << "\n";
        std::cout << "  Bayesian Train MSE = " << autoTrainMSE << "  Val MSE = " << autoValMSE << "\n\n";

        saveResultsToCSV("output_csv/automobile_mle_vs_bayes.csv",
            {"MLE_train", "MLE_val", "Bayes_train", "Bayes_val"},
            {mleTrain, mleVal, autoTrainMSE, autoValMSE});
    }

    std::cout << "Saved: output_csv/wine_mle_vs_bayes.csv\n";
    std::cout << "Saved: output_csv/automobile_mle_vs_bayes.csv\n";

    return 0;
}

#include<iostream>
#include<cmath>
#include<vector>
#include<sstream>
#include<fstream>
#include<string>
#include<thread>
#include<map>  
#include<stdexcept>
#include "basis_functions.h"


double gaussianBasis(
    double x, 
    double mean, 
    double sigma) 
{
    if (sigma <= 0.0) {
        throw std::invalid_argument("Sigma must be > 0 for Gaussian basis.");
    }
    double z = (x - mean) / sigma;
    return std::exp(-0.5 * z * z);
}

double sigmoidalBasis(
    double x, 
    double center, 
    double scale) 
{
    if (scale == 0.0) {
        throw std::invalid_argument("Scale must be non-zero for Sigmoidal basis.");
    }
    double z = (x - center) / scale;
    return 1.0 / (1.0 + std::exp(-z));
}

double polynomialBasis(
    double x, 
    int degree) 
{
    if (degree < 1) {
        throw std::invalid_argument("Polynomial degree must be >= 1.");
    }
    return std::pow(x, degree);
}

std::vector<std::vector<double>> applyPolynomialExpansion(
    const std::vector<std::vector<double>>& data,
    int maxDegree) 
{
    if (data.empty()) {
        return data;
    }
    if (data[0].size() < 2) {
        throw std::invalid_argument("Data must contain at least one feature and one target column.");
    }
    if (maxDegree < 1) {
        throw std::invalid_argument("maxDegree must be >= 1.");
    }
    if (maxDegree == 1) {
        return data;
    }

    std::vector<std::vector<double>> expandedData;
    expandedData.reserve(data.size());

    for (const auto& row : data) {
        std::vector<double> newRow;
        const double target = row.back();

        for (size_t i = 0; i + 1 < row.size(); ++i) {
            for (int degree = 1; degree <= maxDegree; ++degree) {
                newRow.push_back(polynomialBasis(row[i], degree));
            }
        }

        newRow.push_back(target);
        expandedData.push_back(std::move(newRow));
    }

    return expandedData;
}

std::vector<std::vector<double>> applyBasisFunctions(
    const std::vector<std::vector<double>>& data,
    const std::string& basisType,
    double center,
    double scale) 
{
    if (data.empty()) {
        return data;
    }
    if (data[0].size() < 2) {
        throw std::invalid_argument("Data must contain at least one feature and one target column.");
    }

    std::vector<std::vector<double>> transformed = data;
    const size_t featureCount = transformed[0].size() - 1;

    int polynomialDegree = 1;
    if (basisType == "polynomial") {
        polynomialDegree = static_cast<int>(std::round(scale));
        if (polynomialDegree < 1) {
            throw std::invalid_argument("For polynomial basis, scale must represent degree >= 1.");
        }
    }

    for (auto& row : transformed) {
        for (size_t j = 0; j < featureCount; ++j) {
            if (basisType == "gaussian") {
                row[j] = gaussianBasis(row[j], center, scale);
            } else if (basisType == "sigmoidal") {
                row[j] = sigmoidalBasis(row[j], center, scale);
            } else if (basisType == "polynomial") {
                row[j] = polynomialBasis(row[j], polynomialDegree);
            } else if (basisType == "none") {
            } else {
                throw std::invalid_argument("Unknown basisType. Use: gaussian, sigmoidal, polynomial, or none.");
            }
        }
    }

    return transformed;
}

std::vector<std::vector<double>> applyBasisForRegression(
    const std::vector<std::vector<double>>& data,
    const BasisTransformConfig& config) 
{

    if (config.basisType == "gaussian" ||
        config.basisType == "sigmoidal" ||
        config.basisType == "polynomial" ||
        config.basisType == "none") {
        return applyBasisFunctions(data, config.basisType, config.center, config.scale);
    }

    throw std::invalid_argument("Unknown basisType. Use: polynomial, gaussian, sigmoidal, or none.");
}
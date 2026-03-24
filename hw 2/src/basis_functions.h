#ifndef BASIS_FUNCTIONS_H
#define BASIS_FUNCTIONS_H

#include<iostream>
#include<cmath>
#include<vector>
#include<sstream>
#include<fstream>
#include<string>
#include<thread>
#include<map>
#include<stdexcept>


double gaussianBasis(double x, double mean, double sigma);

double sigmoidalBasis(double x, double center, double scale);

double polynomialBasis(double x, int degree);

std::vector<std::vector<double>> applyPolynomialExpansion(
	const std::vector<std::vector<double>>& data,
	int maxDegree);

std::vector<std::vector<double>> applyBasisFunctions(
	const std::vector<std::vector<double>>& data, 
	const std::string& basisType, 
	double center, 
	double scale);

struct BasisTransformConfig {
	std::string basisType = "none";
	int polynomialDegree = 2;
	double center = 0.0;
	double scale = 1.0;
};

std::vector<std::vector<double>> applyBasisForRegression(
	const std::vector<std::vector<double>>& data,
	const BasisTransformConfig& config);

#endif
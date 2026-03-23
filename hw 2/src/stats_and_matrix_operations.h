#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include<iostream>
#include<cmath>
#include<vector>
#include<sstream>
#include<fstream>
#include<string>
#include<thread>
#include<map>

double getMean(std::vector<double>& nums);
double getVariance(std::vector<double>& nums);
double getStandardDeviation(std::vector<double>& nums);

double gaussianBasis(double x, double mean, double sigma);
double sigmoidalBasis(double x, double center, double scale);
double polynomialBasis(double x, int degree);
std::vector<std::vector<double>> applyBasisFunctions(const std::vector<std::vector<double>>& data,
													 const std::string& basisType,
													 double center,
													 double scale);
std::vector<std::vector<double>> applyPolynomialExpansion(const std::vector<std::vector<double>>& data,
                                                          int maxDegree);

std::vector<std::vector<double>> getTranspose(const std::vector<std::vector<double>>& mat);
std::vector<std::vector<double>> getMatrixProduct(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
std::vector<std::vector<double>> getMatrixInverse(const std::vector<std::vector<double>>& matrix);
void getModelMatrices(const std::vector<std::vector<double>>& data, std::vector<std::vector<double>>& X, std::vector<double>& y);

void printMatrix2D(const std::vector<std::vector<double>>& mat);

int monthToNumber(const std::string& month);
int dayToNumber(const std::string& day);

#endif

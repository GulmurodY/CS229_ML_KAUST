#include<iostream>
#include<vector>
#include<sstream>
#include<random>
#include<fstream>
#include<string>
#include<map>
#include "stats_and_matrix_operations.h"
#include "linear_regression_models.h"

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

std::vector<std::vector<double>> loadCSV(const std::string& filename, char delimiter) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return data;
    }
    
    std::string line;
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> row;
        std::string token;
        int column = 0;
        
        while (std::getline(iss, token, delimiter)) {
            try {
                double value = std::stod(token);
                row.push_back(value);
            }
            catch (const std::exception& e) {
                double converted_value = -1;
                
                if (column == 2) {
                    converted_value = static_cast<double>(monthToNumber(token));
                } else if (column == 3) {
                    converted_value = static_cast<double>(dayToNumber(token));
                }
                
                if (converted_value != -1) {
                    row.push_back(converted_value);
                }
            }
            column++;
        }
        
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    
    file.close();
    return data;
}

int main() {
    std::vector<std::string> datasetNames = {"wine/winequality-red"}; //, "wine/winequality-white", "forest fires/forestfires"};
    
    for (const std::string& name : datasetNames) {
        std::string fileName = name;
        for (char &c : fileName) {
            if (c == '/' || c == ' ') c = '_';
        }

        auto rawData = loadCSV("datasets/" + name + ".csv", name[0] == 'w' ? ';' : ',');
        DatasetSplit sets = splitData(rawData, 0.8);
        
        std::vector<double> lambdas_fixed = {0.0};
        std::vector<double> cf_train, cf_val, gd_train, gd_val;
        
        std::vector<double> w_cf = linearRegressionClosedForm(sets.train);
        cf_train.push_back(calculateMSE(sets.train, w_cf));
        cf_val.push_back(calculateMSE(sets.validation, w_cf));
        
        std::vector<double> w_gd = linearRegressionGradientDescent(sets.train);
        gd_train.push_back(calculateMSE(sets.train, w_gd));
        gd_val.push_back(calculateMSE(sets.validation, w_gd));
        
        saveResultsToCSV("experiment/" + fileName + "_method_comparison.csv",
                 {"CF_Train", "CF_Val", "GD_Train", "GD_Val"},
                 {cf_train[0], cf_val[0], gd_train[0], gd_val[0]});
        
        std::vector<double> lambdas = {0, 0.01, 0.1, 1, 10, 100, 1000};
        std::vector<double> reg_train_errors;
        std::vector<double> reg_val_errors;
        
        for (double l : lambdas) {
            std::vector<double> w = linearRegressionClosedForm(sets.train, l);
            reg_train_errors.push_back(calculateMSE(sets.train, w));
            reg_val_errors.push_back(calculateMSE(sets.validation, w));
        }
        
        saveResultsToCSV("experiment/" + fileName + "_regularization_results.csv", lambdas, reg_train_errors, reg_val_errors);
        
        
        std::vector<double> degrees = {1, 2, 3};
        std::vector<double> poly_train_errors;
        std::vector<double> poly_val_errors;
        
        for (double d : degrees) {
            auto trainPoly = expandPolynomial(sets.train, (int)d);
            auto valPoly = expandPolynomial(sets.validation, (int)d);
        
            std::vector<double> w = linearRegressionClosedForm(trainPoly);
            poly_train_errors.push_back(calculateMSE(trainPoly, w));
            poly_val_errors.push_back(calculateMSE(valPoly, w));
        }
        
        saveResultsToCSV("experiment/" + fileName + "_polynomial_results.csv", degrees, poly_train_errors, poly_val_errors);
    }

    return 0;
}
#ifndef CSV_DATA_LOADER_H
#define CSV_DATA_LOADER_H

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <stdexcept>

const std::string titanicPath      = "../data_for_classification/titanic/train.csv";
const std::string irisPath         = "../data_for_classification/Iris/Iris.csv";
const std::string winePath         = "../data_for_regression/wine/winequality-red.csv";
const std::string forestFiresPath  = "../data_for_regression/forest fires/forestfires.csv";
const std::string automobilePath   = "../data_for_regression/automobile/imports-85.data";

struct CSVConfig {
    char delimiter = ',';
    bool hasHeader = true;
    std::set<int> skipColumns;
    std::map<int, std::map<std::string, double>> categoricalMaps;
    int targetColumn = -1;
    double missingDefault = 0.0;
};

std::vector<std::vector<double>> loadCSV(const std::string& filename, const CSVConfig& cfg);
CSVConfig getConfigForDataset(const std::string& filePath);

bool printSummary(const std::string& name,
                  const std::vector<std::vector<double>>& data,
                  std::size_t expectedColumns = 0);
void printStatisticalFeatures(const std::string& name,
                               const std::vector<std::vector<double>>& data);
bool checkDatasetLoading(const std::string& name, const std::string& path,
                         const CSVConfig& cfg, std::size_t expectedColumns);

#endif

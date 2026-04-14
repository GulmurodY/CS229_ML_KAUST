#ifndef CSV_DATA_LOADER_H
#define CSV_DATA_LOADER_H

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <stdexcept>

const std::string titanicPath     = "hw3_data/classification/titanic.csv";
const std::string irisBinaryPath  = "hw3_data/classification/iris_binary.csv";
const std::string xorPath         = "hw3_data/classification/xor_2d.csv";
const std::string circlesPath     = "hw3_data/classification/circles_2d.csv";
const std::string linear2DPath    = "hw3_data/classification/linear_2d.csv";

const std::string winePath        = "hw3_data/regression/wine_red.csv";
const std::string automobilePath  = "hw3_data/regression/automobile.csv";
const std::string forestFiresPath = "hw3_data/regression/forest_fires.csv";
const std::string sin1DPath       = "hw3_data/regression/sin_1d.csv";
const std::string cubic1DPath     = "hw3_data/regression/cubic_1d.csv";
const std::string quadratic1DPath = "hw3_data/regression/quadratic_1d.csv";
const std::string saddle2DPath    = "hw3_data/regression/saddle_2d.csv";
const std::string sinsurf2DPath   = "hw3_data/regression/sinsurf_2d.csv";

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

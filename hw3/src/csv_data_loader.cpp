#include "csv_data_loader.h"
#include "stats_and_matrix_operations.h"
#include <fstream>
#include <sstream>

static std::vector<std::string> parseCSVLine(
    const std::string& line,
    char delimiter)
{
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

std::vector<std::vector<double>> loadCSV(
    const std::string& filename,
    const CSVConfig& cfg)
{
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
        double target    = cfg.missingDefault;
        bool   hasTarget = (cfg.targetColumn >= 0);

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

CSVConfig getConfigForDataset(const std::string& filePath)
{
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

        CSVConfig wineCfg;
        wineCfg.delimiter = ';';
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

bool printSummary(
    const std::string& name,
    const std::vector<std::vector<double>>& data,
    std::size_t expectedColumns)
{
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

void printStatisticalFeatures(
    const std::string& name,
    const std::vector<std::vector<double>>& data)
{
    if (data.empty() || data[0].empty()) return;

    const std::size_t columnCount = data[0].size();
    const std::size_t targetIndex = columnCount - 1;

    std::cout << "Statistical features for " << name << " (features only):\n";
    for (std::size_t col = 0; col < targetIndex; ++col) {
        std::vector<double> values;
        values.reserve(data.size());
        for (const auto& row : data) {
            values.push_back(row[col]);
        }

        double mean     = getMean(values);
        double variance = getVariance(values);
        double stdDev   = getStandardDeviation(values);

        std::cout << "  Feature[" << col << "]"
                  << " mean=" << mean
                  << ", variance=" << variance
                  << ", stdDev=" << stdDev << "\n";
    }
}

bool checkDatasetLoading(
    const std::string& name,
    const std::string& path,
    const CSVConfig& cfg,
    std::size_t expectedColumns)
{
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

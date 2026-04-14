#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include "activations.h"

enum class TaskType {
    Regression,
    Classification
};

class NeuralNetwork {
public:
    NeuralNetwork(int inputDim,
                  int hiddenDim,
                  int outputDim,
                  Activation hiddenActivation,
                  TaskType task);

    std::vector<double> forward(const std::vector<double>& x);

    void train(const std::vector<std::vector<double>>& X,
               const std::vector<std::vector<double>>& Y,
               int epochs,
               double learningRate,
               int batchSize);

    double loss(const std::vector<std::vector<double>>& X,
                const std::vector<std::vector<double>>& Y);

private:
    int inputDim_;
    int hiddenDim_;
    int outputDim_;
    Activation hiddenActivation_;
    TaskType task_;

    std::vector<std::vector<double>> W1_;
    std::vector<double> b1_;
    std::vector<std::vector<double>> W2_;
    std::vector<double> b2_;
};

#endif

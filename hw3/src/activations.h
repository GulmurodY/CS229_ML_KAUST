#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

enum class Activation {
    Sigmoid,
    Tanh,
    ReLU,
    Identity
};

double activate(double z, Activation a);
double activateDerivative(double z, Activation a);

#endif

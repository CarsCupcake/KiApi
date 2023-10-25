package me.carscupcake;

// Code adapted from SebastianLeague
public interface IActivationFunction {
    double activate(double[] inputs, int index);
    double derivative(double[] inputs, int index);
}

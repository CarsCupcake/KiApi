package me.carscupcake;

import me.carscupcake.learn.LayerLearnData;
@SuppressWarnings("unused")
public record TrainingData(double[] input, double[] expected) {
    public void evaluateCost(double[] result, ICost cost, LayerLearnData data, IActivationFunction function) {
        for (int i = 0; i < data.layer().getNodes().length; i++) {
            double costDer = cost.derivative(data.activations()[i], expected[i]);
            double activationDer = function.derivative(result, i);
            data.nodeValues()[i] = costDer * activationDer;
        }
    }
}

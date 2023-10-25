package me.carscupcake;

import me.carscupcake.ki.Node;
import me.carscupcake.ki.WeightedInputs;
import me.carscupcake.learn.LayerLearnData;
import me.carscupcake.learn.NetworkLearnData;

@SuppressWarnings("unused")
public record TrainingData(double[] input, double[] expected) {
    public void evaluateCost(double[] result, ICost cost, LayerLearnData data, IActivationFunction function) {
        for (int i = 0; i < data.layer().getNodes().length; i++) {
            double costDer = cost.derivative(data.activations()[i], expected[i]);
            double activationDer = function.derivative(result, i);
            data.nodeValues()[i] = costDer * activationDer;
        }
    }
    public void evaluateHiddenCost(LayerLearnData data, LayerLearnData prev, IActivationFunction function) {
        int i = 0;
        for (Node n : data.layer().getNodes()) {
            int j = 0;
            double val = 0;
            for (Node pr : prev.layer().getNodes()) {
                WeightedInputs inputs = pr.getConnections()[i];
                double weightedInputDer = inputs.getWeight();
                val += weightedInputDer * prev.nodeValues()[j];
                j++;
            }
            val *= function.derivative(data.inputs(), i);
            data.nodeValues()[i] = val;
            i++;
        }
    }
    public void evaluateCost(double[] result, ICost cost, NetworkLearnData data, IActivationFunction function) {
        evaluateCost(result, cost, data.output(), function);
        for (int i = data.layerData().length - 2; i >= 0; i--) {
            evaluateHiddenCost(data.layerData()[i], data.layerData()[i+1], function);
        }
    }
}

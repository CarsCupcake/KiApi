package me.carscupcake;

import me.carscupcake.ki.Node;
import me.carscupcake.ki.WeightedInputs;
import me.carscupcake.learn.LayerLearnData;
import me.carscupcake.learn.NetworkLearnData;

import java.util.Arrays;

@SuppressWarnings("unused")
public record TrainingData(double[] input, double[] expected) {
    public void evaluateCost(double[] result, ICost cost, LayerLearnData data, IActivationFunction function) {
        for (int i = 0; i < data.layer().getNodes().length; i++) {
            double costDer = cost.derivative(data.activations()[i], expected[i]);
            double activationDer = function.derivative(data.weightedInputs(), i);
            data.nodeValues()[i] = costDer * activationDer;
            if (Double.isNaN(data.nodeValues()[i])) {
                System.out.println("Output node value is NaN!");
                data.nodeValues()[i] = 0;
            }
        }
    }

    public void evaluateHiddenCost(LayerLearnData data, LayerLearnData prev, IActivationFunction function) {
        int i = 0;
        for (Node n : data.layer().getNodes()) {
            int j = 0;
            double val = 0;
            for (Node pr : prev.layer().getNodes()) {
                WeightedInputs inputs = pr.getConnections()[i];
                if (pr != inputs.getPrev()) {
                    for (WeightedInputs in : pr.getConnections())
                        if (in.getPrev() == pr) inputs = in;
                    if (pr != inputs.getPrev()) {
                        System.out.println(Arrays.toString(pr.getConnections()));
                        System.out.println(pr);
                        System.out.println(n);
                        System.out.println(inputs.getPrev());
                        System.exit(-1);
                    }
                }
                double weightedInputDer = inputs.getWeight();
                val += weightedInputDer * prev.nodeValues()[j];
                j++;
            }
            val *= function.derivative(data.inputs(), i);
            data.nodeValues()[i] = val;
            if (Double.isNaN(data.nodeValues()[i])) {
                System.out.println("Hidden node value is NaN!");
                data.nodeValues()[i] = 0;
            }
            i++;
        }
    }

    public void evaluateCost(double[] result, ICost cost, NetworkLearnData data, IActivationFunction function) {
        evaluateCost(result, cost, data.output(), function);
        data.output().update();
        for (int i = data.layerData().length - 1; i > 0; i--) {
            evaluateHiddenCost(data.layerData()[i - 1], data.layerData()[i], function);
            data.layerData()[i].update();
        }
    }
}

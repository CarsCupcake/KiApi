package me.carscupcake.learn;

import me.carscupcake.ki.Layer;

import java.util.Arrays;

public record LayerLearnData(double[] inputs, double[] weightedInputs, double[] activations, double[] nodeValues, Layer layer) {
    public void update() {
        layer.updateGradiants(this);
    }

    @Override
    public String toString() {
        return "LayerLearnData{" +
                "inputs=" + Arrays.toString(inputs) +
                ", weightedInputs=" + Arrays.toString(weightedInputs) +
                ", activations=" + Arrays.toString(activations) +
                ", nodeValues=" + Arrays.toString(nodeValues) +
                '}';
    }
}

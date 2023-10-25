package me.carscupcake.learn;

import me.carscupcake.ki.Layer;

import java.util.Arrays;

public record LayerLearnData(double[] inputs, double[] weightedInputs, double[] activations, double[] nodeValues, Layer layer) {
    public void update() {
        layer.updateGradiants(this);
    }

    @Override
    public String toString() {
        return "[" + layer.toString() + ", " + Arrays.toString(nodeValues) + "]";
    }
}

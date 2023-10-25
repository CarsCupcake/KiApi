package me.carscupcake.learn;

import me.carscupcake.ki.Layer;

public record LayerLearnData(double[] inputs, double[] weightedInputs, double[] activations, double[] nodeValues, Layer layer) {
    public void update() {
        layer.updateGradiants(this);
    }
}

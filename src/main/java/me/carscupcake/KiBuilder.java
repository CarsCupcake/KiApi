package me.carscupcake;

import lombok.Getter;
import me.carscupcake.ki.Input;
import me.carscupcake.ki.Layer;
import me.carscupcake.ki.Output;
import me.carscupcake.util.Assert;

import java.util.ArrayList;
import java.util.List;
@SuppressWarnings("unused")
@Getter
public class KiBuilder {
    private final List<Input> inputs = new ArrayList<>();
    private final List<Output> outputs = new ArrayList<>();
    private final List<Layer> hiddenLayers = new ArrayList<>();
    private IActivationFunction activationFunction = ActivationFunctions.Sigmoid;
    private ICost cost;
    private double regularization, momentum = 0;
    private IActivationFunction outputActivation = ActivationFunctions.Sigmoid;
    public KiBuilder setOutputActivation(IActivationFunction activationFunction) {
        this.outputActivation = activationFunction;
        return this;
    }
    public KiBuilder setRegularization(double d) {
        regularization = d;
        return this;
    }
    public KiBuilder setMomentum(double d) {
        momentum = d;
        return this;
    }
    public KiBuilder setActivationFunction(IActivationFunction functions) {
        activationFunction = functions;
        return this;
    }
    public KiBuilder setCost(ICost cost) {
        this.cost = cost;
        return this;
    }
    public KiBuilder addInput() {
        inputs.add(new Input());
        return this;
    }
    public KiBuilder addInputs(int i) {
        Assert.state(i > 0, "I is not larger than 0");
        for (int j = 0; j++ < i;) {
            addInput();
        }
        return this;
    }
    public KiBuilder addOutput() {
        outputs.add(new Output());
        return this;
    }
    public KiBuilder addOutput(int i) {
        Assert.state(i > 0, "I is not larger than 0");
        for (int j = 0; j++ < i;) {
            addOutput();
        }
        return this;
    }
    public KiBuilder addLayer(int i) {
        hiddenLayers.add(Layer.newLayer(i));
        return this;
    }
    public KiApi build() {
        return new KiApi(this);
    }
}

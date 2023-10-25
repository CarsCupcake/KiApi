package me.carscupcake.ki;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.Getter;
import me.carscupcake.IActivationFunction;
import me.carscupcake.KiApi;
import me.carscupcake.learn.LayerLearnData;

@Getter
public class Layer {
    private final Node[] nodes;
    private double weightDecay = 0;
    public Layer(Node[] nodes) {
        this.nodes = nodes;
    }

    public static Layer newLayer(int j) {
        Node[] n = new Node[j];
        for (int i = 0; i < j; i++)
            n[i] = new Node();
        return new Layer(n);
    }
    public double[] calcOutput(double[] input, IActivationFunction function) {
        double[] f = new double[nodes.length];
        int i = 0;
        int j = 0;
        for (Node n : nodes) {
            double weightInput = n.getBias();
            for (WeightedInputs in : n.getConnections()) {
                weightInput += input[i] * in.getWeight();
                i++;
            }
            f[j] = weightInput;
            j++;
            i = 0;
        }
        //Activation code by SebastianLeague
        double[] activation = new double[nodes.length];
        for (i = 0; i < nodes.length ; i++)
            activation[i] = function.activate(f, i);
        return activation;
    }
    public double[] calcOutput(double[] input, IActivationFunction function, LayerLearnData data) {
        System.arraycopy(input, 0, data.inputs(), 0, input.length);
        double[] f = new double[nodes.length];
        int i = 0;
        int j = 0;
        for (Node n : nodes) {
            double weightInput = n.getBias();
            for (WeightedInputs in : n.getConnections()) {
                weightInput += input[i] * in.getWeight();
                i++;
            }
            data.weightedInputs()[j] = weightInput;
            f[j] = weightInput;
            j++;
            i = 0;
        }
        //Activation code by SebastianLeague
        double[] activation = new double[nodes.length];
        for (i = 0; i < nodes.length ; i++) {
            activation[i] = function.activate(f, i);
            data.activations()[i] = activation[i];
        }
        return activation;
    }
    public void updateGradiants(LayerLearnData data) {
        int i = 0;
        int j = 0;
        for (Node node : nodes) {
            double nodeData = data.nodeValues()[i];
            for (WeightedInputs inputs : node.getConnections()) {
                double derCostWeight = data.inputs()[j] * nodeData;
                inputs.setCostGradiant(inputs.getCostGradiant() + derCostWeight);
                j++;
            }
            j = 0;
            double derCostBias = data.nodeValues()[i];
            node.setCostGradiant(node.getCostGradiant() + derCostBias);
            i++;
        }
    }
    public void addToJson(ArrayNode json) {
        ArrayNode array = KiApi.factory.arrayNode();
        for (Node n : nodes)
            array.add(n.toJson());
        json.add(array);
    }
    public void learn(double learnRate, double regularization, double momentum) {
        weightDecay = (1 - regularization * momentum);
        for (Node node : nodes) {
            for (WeightedInputs input : node.getConnections()) {
                double weight = input.getWeight();
                double velocity = input.getVelocity() * momentum - input.getCostGradiant() * learnRate;
                input.setWeight(weight * weightDecay + velocity);
                input.setCostGradiant(0);
            }
            double velocity = node.getVelocity() * momentum - node.getCostGradiant() * learnRate;
            node.setVelocity(velocity);
            node.setBias(node.getBias() + velocity);
            node.setCostGradiant(0);
        }
    }
    public static Layer from(ArrayNode object, Layer prev) {
        Node[] nodes = new Node[object.size()];
        int i = 0;
        for (JsonNode array : object) {
            nodes[i] = Node.from((ObjectNode) array, prev);
            i++;
        }
        return new Layer(nodes);
    }
}

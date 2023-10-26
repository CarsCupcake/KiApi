package me.carscupcake.ki;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.Getter;
import me.carscupcake.IActivationFunction;
import me.carscupcake.KiApi;
import me.carscupcake.learn.LayerLearnData;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

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
        int j = 0;
        for (Node n : nodes) {
            int i = 0;
            double weightInput = n.getBias();
            for (WeightedInputs in : n.getConnections()) {
                weightInput += input[i] * in.getWeight();
                i++;
            }
            if (Double.isNaN(weightInput)) {
                System.out.println("Weight Input is NaN! (40)");
                weightInput = 0;
            }
            f[j] = weightInput;
            j++;
        }
        //Activation code by SebastianLeague
        double[] activation = new double[nodes.length];
        for (int i = 0; i < nodes.length ; i++) {
            activation[i] = function.activate(f, i);
            if (Double.isNaN(activation[i])) {
                System.out.println("Activation is NaN! (51)");
                activation[i] = 0;
            }
        }
        return activation;
    }
    private static final Set<Double> data = new HashSet<>();
    public double[] calcOutput(double[] input, IActivationFunction function, LayerLearnData data) {
        System.arraycopy(input, 0, data.inputs(), 0, input.length);
        int j = 0;
        for (Node n : nodes) {
            int i = 0;
            double weightInput = n.getBias();
            for (WeightedInputs in : n.getConnections()) {
                weightInput += input[i] * in.getWeight();
                i++;
            }
            if (Double.isNaN(weightInput)) {
                System.out.println("Weight Input is NaN!");
                weightInput = 0;
            }
            data.weightedInputs()[j] = weightInput;
            j++;
        }
        //Activation code by SebastianLeague
        for (int i = 0; i < data.activations().length ; i++) {
            data.activations()[i] = function.activate(data.weightedInputs(), i);
            if (Double.isNaN(data.activations()[i])) {
                System.out.println("Activation is NaN!");
                data.activations()[i] = 0;
            }
        }
        return data.activations();
    }
    public void updateGradiants(LayerLearnData data) {
        int i = 0;
        int j = 0;
        for (Node node : nodes) {
            double nodeData = data.nodeValues()[i];
            for (WeightedInputs inputs : node.getConnections()) {
                double derCostWeight = data.inputs()[j] * nodeData;
                if (Double.isNaN(derCostWeight)) {
                    System.out.println("derCostWeight is NaN!");
                    derCostWeight = 0;
                }
                inputs.setCostGradiant(inputs.getCostGradiant() + derCostWeight);
                j++;
            }
            j = 0;
            double derCostBias = data.nodeValues()[i];
            if (Double.isNaN(derCostBias)) {
                System.out.println("derCostBias is NaN!");
                derCostBias = 0;
            }
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
    public void applyGradiants(double learnRate, double regularization, double momentum) {
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
            if (Double.isNaN(velocity)) {
                velocity = 0;
            }
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

    @Override
    public String toString() {
        return "Layer{" +
                "nodes=" + Arrays.toString(nodes) +
                ", weightDecay=" + weightDecay +
                '}';
    }
}

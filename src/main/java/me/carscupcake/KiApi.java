package me.carscupcake;

import com.fasterxml.jackson.core.util.DefaultIndenter;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.Getter;
import me.carscupcake.ki.*;
import me.carscupcake.learn.LayerLearnData;
import me.carscupcake.learn.NetworkLearnData;
import me.carscupcake.util.Assert;

import java.io.*;
import java.util.Random;

@SuppressWarnings("unused")
@Getter
public class KiApi {
    public static JsonNodeFactory factory = new JsonNodeFactory(false);
    private final Layer inputs;
    private final Layer outputs;
    private final Layer[] hiddenLayers;
    private final IActivationFunction function;
    private final ICost cost;
    private final double momentum;
    private final double regularization;
    private final IActivationFunction outputActivation;

    KiApi(KiBuilder builder) {
        inputs = new Layer(builder.getInputs().toArray(new Input[0]));
        outputs = new Layer(builder.getOutputs().toArray(new Output[0]));
        hiddenLayers = builder.getHiddenLayers().toArray(new Layer[0]);
        this.function = builder.getActivationFunction();
        this.cost = builder.getCost();
        this.momentum = builder.getMomentum();
        this.regularization = builder.getRegularization();
        Assert.notNull(cost, "Cost is null!");
        outputActivation = builder.getOutputActivation();
        generateConnections();
    }

    public KiApi(File f, ICost cost, IActivationFunction function, IActivationFunction outputActivation) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(f));

        ObjectNode obj = new ObjectMapper().readValue(f, ObjectNode.class);
        ArrayNode array = (ArrayNode) obj.get("nodeValues");
        momentum = obj.get("momentum").asDouble();
        regularization = obj.get("regularization").asDouble();
        int layers = array.size();
        hiddenLayers = new Layer[array.size() - 2];
        int i = -1;
        Layer inputLayer = null;
        Layer outputLayer = null;
        int size = array.size();
        Layer last = null;
        for (JsonNode element : array) {
            i++;
            if (i == 0) {
                Layer l = Layer.from((ArrayNode) element, null);
                last = l;
                inputLayer = l;
                continue;
            }
            Layer l = Layer.from((ArrayNode) element, last);
            last = l;
            if (i == size - 1) outputLayer = l;
            else {
                hiddenLayers[i - 1] = l;
            }
        }
        inputs = inputLayer;
        outputs = outputLayer;
        Assert.notNull(cost, "Cost is null!");
        this.cost = cost;
        Assert.notNull(function, "Cost is null!");
        this.function = function;
        this.outputActivation = outputActivation;
    }

    public NetworkLearnData makeLearnData() {
        LayerLearnData[] data = new LayerLearnData[hiddenLayers.length + 2];
        Layer last = null;
        for (int i = 0; i < data.length; i++) {
            Layer l = (i == 0) ? inputs : ((i == data.length - 1) ? outputs : hiddenLayers[i - 1]);
            data[i] = new LayerLearnData(new double[lastNodeSize(last)], new double[(last == null) ? inputs.getNodes().length : last.getNodes().length * l.getNodes().length], new double[l.getNodes().length], new double[l.getNodes().length], l);
            last = l;
        }
        return new NetworkLearnData(data);
    }

    private int lastNodeSize(Layer last) {
        return (last == null) ? inputs.getNodes().length : last.getNodes().length;
    }

    public double[] ask(double[] input) {
        double[] values = input;
        int i = 0;
        for (Layer l : hiddenLayers) {
            i++;
            values = l.calcOutput(values, function);
        }
        return outputs.calcOutput(values, outputActivation);
    }

    private double[] learn(double[] input, NetworkLearnData networkData) {
        double[] values = input;
        int i = 0;
        System.arraycopy(input, 0, networkData.layerData()[0].inputs(), 0, input.length);
        for (Layer l : hiddenLayers) {
            i++;
            values = l.calcOutput(values, function, networkData.layerData()[i]);
        }
        return outputs.calcOutput(values, outputActivation, networkData.output());
    }

    /**
     * Trains the ai
     * @param data The training data
     * @param learnRate a modifier for the learning rate
     */
    public void train(TrainingData[] data, double learnRate) {
        int i = 0;
        int total = 0;
        for (TrainingData d : data) {
            NetworkLearnData networkData = makeLearnData();
            try {
                double[] out = learn(d.input(), networkData);

                d.evaluateCost(out, KiApi.this.cost, networkData, function);
                networkData.update();
                if (i >= 500) {
                    System.gc();
                    i = 0;
                }
            } catch (Exception e) {
                e.printStackTrace(System.err);
            }
            i++;
        }
        for (LayerLearnData d : makeLearnData().layerData())
            d.layer().applyGradiants(learnRate / data.length, regularization, momentum);
    }

    public void save(File file) {
        ArrayNode array = factory.arrayNode();
        inputs.addToJson(array);
        for (Layer l : hiddenLayers)
            l.addToJson(array);
        outputs.addToJson(array);
        ObjectNode object = factory.objectNode();
        object.set("nodeValues", array);
        object.put("momentum", momentum);
        object.put("regularization", regularization);
        try {
            ObjectWriter objectWriter = new ObjectMapper().writer(new DefaultPrettyPrinter().withObjectIndenter(new DefaultIndenter().withLinefeed("\n")));
            objectWriter.writeValue(file, object);
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }
    }

    private void generateConnections() {
        int i = 0;
        for (Layer l : hiddenLayers) {
            Layer prev = (i == 0) ? inputs : hiddenLayers[i - 1];
            makeNodes(prev, l);
            i++;
        }
        makeNodes((hiddenLayers.length != 0) ? hiddenLayers[hiddenLayers.length - 1] : inputs, outputs);
    }

    private void makeNodes(Layer prev, Layer l) {
        Random r = new Random();
        for (Node n : l.getNodes()) {
            int ii = 0;
            WeightedInputs[] in = new WeightedInputs[prev.getNodes().length];
            for (Node previous : prev.getNodes()) {
                in[ii] = new WeightedInputs(n, randomInNormalDistrubution(r, 0, 1) / Math.sqrt(prev.getNodes().length));
                ii++;
            }
            n.setConnections(in);
        }
    }
    private double randomInNormalDistrubution(Random random, double mean, double standartDerivation) {
        double x1 = 1d - random.nextDouble();
        double x2 = 1d - random.nextDouble();
        double t = -2d * Math.log(x1) * Math.cos(2d * Math.PI * x2);
        double y1;
        if (t < 0) {
            y1 = -Math.sqrt(t * -1);
        } else y1 = Math.sqrt(t);
        return y1 * standartDerivation + mean;
    }
}

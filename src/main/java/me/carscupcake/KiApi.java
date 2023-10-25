package me.carscupcake;

import com.fasterxml.jackson.core.util.DefaultIndenter;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import me.carscupcake.ki.*;
import me.carscupcake.learn.LayerLearnData;
import me.carscupcake.learn.NetworkLearnData;
import me.carscupcake.util.Assert;

import java.io.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

@SuppressWarnings("unused")
public class KiApi {
    public static JsonNodeFactory factory = new JsonNodeFactory(false);
    private final Layer inputs;
    private final Layer outputs;
    private final Layer[] hiddenLayers;
    private final IActivationFunction function;
    private final ICost cost;
    private final NetworkLearnData networkData;
    private final double momentum;
    private final double regularization;

    KiApi(KiBuilder builder) {
        inputs = new Layer(builder.getInputs().toArray(new Input[0]));
        outputs = new Layer(builder.getOutputs().toArray(new Output[0]));
        hiddenLayers = builder.getHiddenLayers().toArray(new Layer[0]);
        this.function = builder.getActivationFunction();
        this.cost = builder.getCost();
        this.momentum = builder.getMomentum();
        this.regularization = builder.getRegularization();
        Assert.notNull(cost, "Cost is null!");
        generateConnections();
        LayerLearnData[] data = new LayerLearnData[hiddenLayers.length + 2];
        Layer last = null;
        for (int i = 0; i < data.length; i++) {
            Layer l = (i == 0) ? inputs : ((i == data.length - 1) ? outputs : hiddenLayers[i - 1]);
            data[i] = new LayerLearnData(new double[lastNodeSize(last)], new double[(last == null) ? inputs.getNodes().length : last.getNodes().length * l.getNodes().length],
                    new double[l.getNodes().length], new double[l.getNodes().length], l);
            last = l;
        }
        networkData = new NetworkLearnData(data);
    }
    public KiApi(File f, ICost cost, IActivationFunction function) throws IOException {
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
        generateConnections();
        LayerLearnData[] data = new LayerLearnData[hiddenLayers.length + 2];
        last = null;
        for (i = 0; i < data.length; i++) {
            Layer l = (i == 0) ? inputs : ((i == data.length - 1) ? outputs : hiddenLayers[i - 1]);
            data[i] = new LayerLearnData(new double[lastNodeSize(last)], new double[(last == null) ? inputs.getNodes().length : last.getNodes().length * l.getNodes().length],
                    new double[l.getNodes().length], new double[l.getNodes().length], l);
            last = l;
        }
        networkData = new NetworkLearnData(data);
    }

    private int lastNodeSize(Layer last) {
        return (last == null) ? 0 : last.getNodes().length;
    }

    public double[] ask(double[] input) {
        double[] values = input;
        int i = 0;
        for (Layer l : hiddenLayers) {
            i++;
            values = l.calcOutput(values, (i < hiddenLayers.length) ? hiddenLayers[i].getNodes().length : outputs.getNodes().length, function);
        }
        return outputs.calcOutput(values, (i < hiddenLayers.length) ? hiddenLayers.length : outputs.getNodes().length, function);
    }

    private double[] learn(double[] input) {
        double[] values = input;
        int i = 0;
        for (Layer l : hiddenLayers) {
            i++;
            values = l.calcOutput(values, (i == 1) ? inputs.getNodes().length : hiddenLayers[i - 1].getNodes().length, function, networkData.layerData()[i]);
        }
        i++;
        return outputs.calcOutput(values, (i < hiddenLayers.length) ? hiddenLayers.length : outputs.getNodes().length, function, networkData.layerData()[i]);
    }

    public void train(TrainingData[] data, double learnRate) {
        AtomicReference<Double> costMini = new AtomicReference<>(Double.MAX_VALUE);
        int started = 0;
        AtomicInteger finished = new AtomicInteger();
        for (TrainingData d : data) {
            started++;
            Thread.ofVirtual().factory().newThread(() -> {
                double[] out = learn(d.input());
                d.evaluateCost(out, KiApi.this.cost, networkData.output(), function);
                networkData.update();
                synchronized (finished) {
                    finished.getAndIncrement();
                }
            }).start();
        }
        while (true) {
            synchronized (finished) {
                if (finished.get() == started) break;
            }
        }
        for (LayerLearnData d : networkData.layerData()) d.layer().learn(learnRate, regularization, momentum);
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
            ObjectWriter objectWriter = new ObjectMapper().writer(new DefaultPrettyPrinter()
                    .withObjectIndenter(new DefaultIndenter().withLinefeed("\n")));
            objectWriter.writeValue(file, object);
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }
    }

    private void generateConnections() {
        int i = 0;
        for (Layer l : hiddenLayers) {
            Layer prev = (i == 0) ? inputs : hiddenLayers[i - 1];
            for (Node n : l.getNodes()) {
                int ii = 0;
                WeightedInputs[] in = new WeightedInputs[prev.getNodes().length];
                for (Node previous : prev.getNodes()) {
                    in[ii] = new WeightedInputs(n);
                    ii++;
                }
                n.setConnections(in);
            }
            i++;
        }
    }
}

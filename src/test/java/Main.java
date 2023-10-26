import me.carscupcake.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.*;
import java.util.*;

public class Main {
    private static KiApi kiApi;
    public static void main(String[] args) throws IOException {
        File trainFolder = new File("D:\\Java\\Code\\Netowrk\\trainingSet");
        File testFolder = new File("D:\\Java\\Code\\Netowrk\\trainingData");
        File data = new File(trainFolder, "data.json");
        new Thread(() -> {
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            while (true) {
                try {
                    String line = reader.readLine();
                    if (line.equals("exit")) {
                        synchronized (kiApi){
                            kiApi.save(data);
                            System.exit(0);
                        }
                    }
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        }).start();
        if (!data.exists() || new BufferedReader(new FileReader(data)).lines().findAny().isEmpty()) {
            data.createNewFile();
            kiApi = new KiBuilder().setCost(CostFunctions.CrossEntropy).setOutputActivation(ActivationFunctions.Softmax).addInputs(784).addLayer(100).addOutput(10)
                    .setRegularization(0).setMomentum(0.9).setActivationFunction(ActivationFunctions.ReLU).build();
        } else {
            kiApi = new KiApi(data, CostFunctions.CrossEntropy, ActivationFunctions.ReLU, ActivationFunctions.Softmax);
            System.out.println("Succsessfully loaded data from " + data.getAbsolutePath());
        }
        System.out.println("AI with " + kiApi.getInputs().getNodes().length + " inputs " + kiApi.getHiddenLayers().length + " layers and " + kiApi.getOutputs().getNodes().length);
        if (!trainFolder.isDirectory())
            throw new IllegalStateException(trainFolder.getAbsolutePath() + " is not a folder!");
        if (!trainFolder.exists())
            throw new IllegalStateException(trainFolder.getAbsolutePath() + " does not exits!");
        Set<TrainingData> trainingData = toData(trainFolder);
        Set<TrainingData> testData = toData(testFolder);
        System.out.println("Start training!");
        while (true){
            train(trainingData);
            System.out.println("Trained random shouffle set");
            double d = test(testData);
            if (d > 0.9)
                break;
            else System.out.println("Test result: " + String.format("%.2f", d * 100) + "%");
        }
        System.out.println("Test bestanden!");
        kiApi.save(data);
        System.exit(0);
        //while (true) {
        //    train(trainingData);
        //    System.out.println(test(trainingData) * 100);
        //}
        /*System.out.println(test(trainingData) * 100);
        BufferedImage in = ImageIO.read(new File("D:\\Java\\Code\\Netowrk\\trainingData", "test.png"));
        double[] res = kiApi.ask(getColorIntensity(in));
        System.out.println(Expectation.getExpectation(res));
        System.out.println(Arrays.toString(res));
        kiApi.save(data);
        System.exit(0);*/
    }
    private static Set<TrainingData> toData(File folder) {
        File zero = new File(folder, "0");
        TrainingData[] zeroTrainingData = makeTrainingData(zero, Expectation.Zero.getData());
        System.out.println("Load " + zeroTrainingData.length + " zero's");
        File one = new File(folder, "1");
        TrainingData[] oneTrainingData = makeTrainingData(one, Expectation.One.getData());
        System.out.println("Load " + oneTrainingData.length + " one's");
        File two = new File(folder, "2");
        TrainingData[] twoTrainingData = makeTrainingData(two, Expectation.Two.getData());
        System.out.println("Load " + twoTrainingData.length + " two's");
        File three = new File(folder, "3");
        TrainingData[] threeTrainingData = makeTrainingData(three, Expectation.Three.getData());
        System.out.println("Load " + threeTrainingData.length + " three's");
        File four = new File(folder, "4");
        TrainingData[] fourTrainingData = makeTrainingData(four, Expectation.Four.getData());
        System.out.println("Load " + fourTrainingData.length + " four's");
        File five = new File(folder, "5");
        TrainingData[] fiveTrainingData = makeTrainingData(five, Expectation.Five.getData());
        System.out.println("Load " + fiveTrainingData.length + " five's");
        File six = new File(folder, "6");
        TrainingData[] sixTrainingData = makeTrainingData(six, Expectation.Six.getData());
        System.out.println("Load " + sixTrainingData.length + " six's");
        File seven = new File(folder, "7");
        TrainingData[] sevenTrainingData = makeTrainingData(seven, Expectation.Seven.getData());
        System.out.println("Load " + sevenTrainingData.length + " seven's");
        File eight = new File(folder, "8");
        TrainingData[] eightTrainingData = makeTrainingData(eight, Expectation.Eight.getData());
        System.out.println("Load " + eightTrainingData.length + " eight's");
        File nine = new File(folder, "9");
        TrainingData[] nineTrainingData = makeTrainingData(nine, Expectation.Nine.getData());
        System.out.println("Load " + nineTrainingData.length + " ninie's");
        Set<TrainingData> trainingData = new HashSet<>(Set.of(zeroTrainingData));
        trainingData.addAll(Set.of(oneTrainingData));
        trainingData.addAll(Set.of(twoTrainingData));
        trainingData.addAll(Set.of(threeTrainingData));
        trainingData.addAll(Set.of(fourTrainingData));
        trainingData.addAll(Set.of(fiveTrainingData));
        trainingData.addAll(Set.of(sevenTrainingData));
        trainingData.addAll(Set.of(sixTrainingData));
        trainingData.addAll(Set.of(eightTrainingData));
        trainingData.addAll(Set.of(nineTrainingData));
        return trainingData;
    }
    private static final int TRAINING_CYCLES = 5;
    private static void train(Set<TrainingData> trainingData) {
        for (int i = 0; i < TRAINING_CYCLES; i++) {
            kiApi.train(trainingData.toArray(new TrainingData[0]), 1);
        }
    }
    private static double test(Set<TrainingData> trainingData) {
        int i = 0;
        Map<Expectation, Integer> total = new HashMap<>();
        for (TrainingData data : trainingData) {
            Expectation expectation = Expectation.getExpectation(data.expected());
            total.put(expectation, total.getOrDefault(expectation, 0) + 1);
        }
        Map<Expectation, Integer> correct = new HashMap<>();
        for (TrainingData data : trainingData) {
            double[] res = kiApi.ask(data.input());
            if (Expectation.getExpectation(res) == Expectation.getExpectation(data.expected())) {
                i++;
                Expectation expectation = Expectation.getExpectation(data.expected());
                correct.put(expectation, correct.getOrDefault(expectation, 0) + 1);
            } else {
                System.out.println("Expected: " + Expectation.getExpectation(data.expected()) + " result: " + toString(res));
            }
        }
        for (Expectation expectation : Expectation.values()) {
            if (!total.containsKey(expectation)) continue;
            System.out.println("Guessed " + expectation + ": " + String.format("%.2f", 100 * ( (double) correct.getOrDefault(expectation, 0) / (double) total.get(expectation))) + "%");
        }
        System.out.println("Correct guesses: " + i);
        return ((double) i) / ((double) trainingData.size());
    }

    private static String toString(double[] array) {
        StringBuilder builder = new StringBuilder("[" + String.format("%.3f", array[0]));
        for (double d : array) {
            builder.append(",").append(String.format("%.3f", d));
        }
        return builder.append("]").toString();
    }

    private static TrainingData[] makeTrainingData(File folder, double[] expected) {
        TrainingData[] data = new TrainingData[Objects.requireNonNull(folder.listFiles()).length];
        int i = 0;
        for (File file : Objects.requireNonNull(folder.listFiles())) {
            try {
                BufferedImage in = ImageIO.read(file);
                data[i] = new TrainingData(getColorIntensity(in), expected);
            }catch (Exception e) {
                throw new RuntimeException(e);
            }
            i++;
        }
        return data;
    }
    private static double[] getColorIntensity(BufferedImage image) {

        final byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        final int width = image.getWidth();
        final int height = image.getHeight();
        final boolean hasAlphaChannel = image.getAlphaRaster() != null;

        double[] result = new double[784];
        int i = 0;
        if (hasAlphaChannel) {
            final int pixelLength = 4;
            for (int pixel = 0; pixel + 3 < pixels.length; pixel += pixelLength) {
                result[i] = ((double) ((int) pixels[pixel + 1] & 0xff)) / 255d;
                if (Double.isNaN(result[i])) throw new NumberFormatException("NaN!");
                i++;
            }
        } else {
            final int pixelLength = 3;
            for (int pixel = 0; pixel + 2 < pixels.length; pixel += pixelLength) {
                result[i] = ((double) ((int) pixels[pixel] & 0xff)) / 255d;
                if (Double.isNaN(result[i])) throw new NumberFormatException("NaN!");
                i++;
            }
        }

        return result;
    }
}
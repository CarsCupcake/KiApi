import me.carscupcake.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.*;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

public class Main {
    private static KiApi kiApi;
    public static void main(String[] args) throws IOException {
        File folder = new File("D:\\Java\\Code\\Netowrk\\trainingData");
        File data = new File(folder, "data.json");
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
            kiApi = new KiBuilder().setCost(CostFunctions.MeanSquareError).addInputs(784).addLayer(500).addLayer(300).addLayer(100).addOutput(10)
                    .setRegularization(1).setMomentum(0.5).setActivationFunction(ActivationFunctions.Sigmoid).build();
        } else kiApi = new KiApi(data, CostFunctions.MeanSquareError, ActivationFunctions.Sigmoid);

        if (!folder.isDirectory())
            throw new IllegalStateException(folder.getAbsolutePath() + " is not a folder!");
        if (!folder.exists())
            throw new IllegalStateException(folder.getAbsolutePath() + " does not exits!");
        System.out.println("Reading files from " + folder.getAbsolutePath());
        File zero = new File(folder, "0");
        TrainingData[] zeroTrainingData = makeTrainingData(zero, Expectation.Zero.getData());
        File one = new File(folder, "1");
        TrainingData[] oneTrainingData = makeTrainingData(one, Expectation.One.getData());
        File two = new File(folder, "2");
        TrainingData[] twoTrainingData = makeTrainingData(two, Expectation.Two.getData());
        File three = new File(folder, "3");
        TrainingData[] threeTrainingData = makeTrainingData(three, Expectation.Three.getData());
        File four = new File(folder, "4");
        TrainingData[] fourTrainingData = makeTrainingData(four, Expectation.Four.getData());
        File five = new File(folder, "5");
        TrainingData[] fiveTrainingData = makeTrainingData(five, Expectation.Five.getData());
        File six = new File(folder, "6");
        TrainingData[] sixTrainingData = makeTrainingData(six, Expectation.Six.getData());
        File seven = new File(folder, "7");
        TrainingData[] sevenTrainingData = makeTrainingData(seven, Expectation.Seven.getData());
        File eight = new File(folder, "8");
        TrainingData[] eightTrainingData = makeTrainingData(eight, Expectation.Eight.getData());
        File nine = new File(folder, "9");
        TrainingData[] nineTrainingData = makeTrainingData(nine, Expectation.Nine.getData());
        Set<TrainingData> trainingData = new HashSet<>(Set.of(zeroTrainingData));
        trainingData.addAll(Set.of(oneTrainingData));
        trainingData.addAll(Set.of(twoTrainingData));
        trainingData.addAll(Set.of(threeTrainingData));
        trainingData.addAll(Set.of(fourTrainingData));
        trainingData.addAll(Set.of(sevenTrainingData));
        trainingData.addAll(Set.of(sixTrainingData));
        trainingData.addAll(Set.of(eightTrainingData));
        trainingData.addAll(Set.of(nineTrainingData));
        while (true){
            train(Set.of(oneTrainingData));
            System.out.println("Trained set 1");
            train(Set.of(twoTrainingData));
            System.out.println("Trained set 2");
            train(Set.of(threeTrainingData));
            System.out.println("Trained set 3");
            train(Set.of(fourTrainingData));
            System.out.println("Trained set 4");
            train(Set.of(fiveTrainingData));
            System.out.println("Trained set 5");
            train(Set.of(sevenTrainingData));
            System.out.println("Trained set 6");
            train(Set.of(sixTrainingData));
            System.out.println("Trained set 7");
            train(Set.of(eightTrainingData));
            System.out.println("Trained set 8");
            train(Set.of(nineTrainingData));
            System.out.println("Trained set 9");
            train(trainingData);
            System.out.println("Trained random shouffle set");
            double d = test(trainingData);
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
    private static final int TRAINING_CYCLES = 10;
    private static void train(Set<TrainingData> trainingData) {
        for (int i = 0; i < TRAINING_CYCLES; i++) {
            kiApi.train(trainingData.toArray(new TrainingData[0]), 2);
        }
    }
    private static double test(Set<TrainingData> trainingData) {
        int i = 0;
        for (TrainingData data : trainingData)
            if (Expectation.getExpectation(kiApi.ask(data.input())) == Expectation.getExpectation(data.expected())) i++;
        return ((double) i) / ((double) trainingData.size());
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
                result[i] = ((int) pixels[pixel + 1] & 0xff) / 255d;
                i++;
            }
        } else {
            final int pixelLength = 3;
            for (int pixel = 0; pixel + 2 < pixels.length; pixel += pixelLength) {
                result[i] = ((int) pixels[pixel] & 0xff) / 255d;
                i++;
            }
        }

        return result;
    }
}
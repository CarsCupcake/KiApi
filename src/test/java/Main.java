import me.carscupcake.*;

import java.io.File;
import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {
        KiApi api = new KiBuilder().setCost(CostFunctions.MeanSquareError).addInputs(2).addLayer(2).addOutput(1).build();
        TrainingData[] trainingData = {new TrainingData(new double[]{1d, 0.5d}, new double[]{0D}), new TrainingData(new double[]{1d, 1d}, new double[]{1d})};
        api.train(trainingData, 0.1);
        File f = new File("data.json");
        f.createNewFile();
        api.save(f);
        api = new KiApi(f, CostFunctions.MeanSquareError, ActivationFunctions.Sigmoid);
    }
}

package me.carscupcake;
@SuppressWarnings("unused")
public enum CostFunctions implements ICost {
    MeanSquareError {
        @Override
        public double function(double[] prediction, double[] expection) {
            double cost = 0;
            for (int i = 0; i < prediction.length; i++)
                cost += Math.pow(prediction[i] - expection[i], 2);
            return cost;
        }

        @Override
        public double derivative(double prediction, double expection) {
            return prediction - expection;
        }
    },
    CrossEntropy {
        @Override
        public double function(double[] prediction, double[] expection) {
            double cost = 0;
            for (int i = 0; i < prediction.length; i++) {
                double x = prediction[i];
                double y = expection[i];
                double v = (y == 1) ? - Math.log(x) : - Math.log(1 - x);
                cost += (Double.isNaN(v)) ? 0 : v;
            }
            return cost;
        }

        @Override
        public double derivative(double prediction, double expection) {
            if (prediction == 0 || expection == 0)
                return 0;
            return (-prediction + expection) / (prediction * (prediction - 1));
        }
    }
}

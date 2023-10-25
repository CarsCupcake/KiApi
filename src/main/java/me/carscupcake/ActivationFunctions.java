package me.carscupcake;
// Code adapted from SebastianLeague
@SuppressWarnings("unused")
public enum ActivationFunctions implements IActivationFunction{
    Sigmoid {
        @Override
        public double activate(double[] inputs, int index) {
            return 1d / (1d + Math.exp(-inputs[index]));
        }

        @Override
        public double derivative(double[] inputs, int index) {
            double a = activate(inputs, index);
            return a * (1 - a);
        }
    },
    TanH {
        @Override
        public double activate(double[] inputs, int index) {
            double e2 = Math.exp(2 * inputs[index]);
            return (e2 - 1) / (e2 + 1);
        }

        @Override
        public double derivative(double[] inputs, int index) {
            return 1 - Math.pow(activate(inputs, index), 2);
        }
    },
    ReLU {
        @Override
        public double activate(double[] inputs, int index) {
            return Math.max(0, inputs[index]);
        }

        @Override
        public double derivative(double[] inputs, int index) {
            return (inputs[index] > 0) ? 1 : 0;
        }

    },
    SiLU {
        @Override
        public double activate(double[] inputs, int index) {
            return inputs[index] / (1 + Math.exp(-inputs[index]));
        }

        @Override
        public double derivative(double[] inputs, int index) {
            double sig = Sigmoid.activate(inputs, index);
            return inputs[index] * sig * (1 - sig) + sig;
        }

    },
    Softmax {
        @Override
        public double activate(double[] inputs, int index) {
            double expSum = 0;
            for (double d : inputs)
                expSum += Math.exp(d);
            return Math.exp(inputs[index]) / expSum;
        }

        @Override
        public double derivative(double[] inputs, int index) {
            double expSum = 0;
            for (double d : inputs)
                expSum += Math.exp(d);
            double ex = Math.exp(inputs[index]);
            return (ex * expSum - Math.pow(ex, 2)) / Math.pow(expSum, 2);
        }

    }
}

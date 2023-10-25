package me.carscupcake;
@SuppressWarnings("unused")
public interface ICost {
    double function(double[] prediction, double[] expection);
    double derivative(double prediction, double expection);
}

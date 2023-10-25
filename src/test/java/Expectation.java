import lombok.Getter;

@Getter
public enum Expectation {
    Zero(new double[]{1,0,0,0,0,0,0,0,0,0}),
    One(new double[]{0,1,0,0,0,0,0,0,0,0}),
    Two(new double[]{0,0,1,0,0,0,0,0,0,0}),
    Three(new double[]{0,0,0,1,0,0,0,0,0,0}),
    Four(new double[]{0,0,0,0,1,0,0,0,0,0}),
    Five(new double[]{0,0,0,0,0,1,0,0,0,0}),
    Six(new double[]{0,0,0,0,0,1,0,0,0,0}),
    Seven(new double[]{0,0,0,0,0,0,0,1,0,0}),
    Eight(new double[]{0,0,0,0,0,0,0,0,1,0}),
    Nine(new double[]{0,0,0,0,0,0,0,0,0,1});
    private final double[] data;
    Expectation(double[] data) {
        this.data = data;
    }
    public static Expectation getExpectation(double[] in) {
        double last = 0;
        int index = 0;
        for (int i = 0; i < in.length ;i++) {
            if (in[i] > last) {
                index = i;
                last = in[i];
            }
        }
        return Expectation.values()[index];
    }
}

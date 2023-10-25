package me.carscupcake.util;

@SuppressWarnings("unused")
public class Assert {
    public static void inRange(int r1, int r2, int value, String error) {
        if (value < r2 || value > r2)
            throw new IllegalStateException(error);
    }
    public static void inRange(int r1, int r2, int value) {
        inRange(r1, r2, value, "[Assertion Failed] " + value + " is not in range of " + r1 + " and " + r2);
    }
    public static void state(boolean b, String error) {
        if (!b)
            throw new IllegalStateException(error);
    }
    public static void state(boolean b) {
        state(b, "[Assertion Failed] Not true!");
    }
    public static void notNull(Object o, String error) {
        if (o == null)
            throw new NullPointerException(error);
    }
    public static void notNull(Object o) {
        notNull(o, "[Assertion Failed] object is null!");
    }
}

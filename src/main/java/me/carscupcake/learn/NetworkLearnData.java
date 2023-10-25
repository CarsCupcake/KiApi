package me.carscupcake.learn;

import java.util.Arrays;

public record NetworkLearnData(LayerLearnData[] layerData) {
    public synchronized void update() {
        int i = -1;
        for (LayerLearnData data : layerData) {
            i++;
            if (i == 0) continue;
            data.update();
        }
    }
    public LayerLearnData output() {
        return layerData[layerData.length - 1];
    }

    @Override
    public String toString() {
        return "NetworkLearnData{" +
                "layerData=" + Arrays.toString(layerData) +
                '}';
    }
}

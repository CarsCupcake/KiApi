package me.carscupcake.learn;

public record NetworkLearnData(LayerLearnData[] layerData) {
    public synchronized void update() {
        for (LayerLearnData data : layerData)
            data.update();
    }
    public LayerLearnData output() {
        return layerData[layerData.length - 1];
    }
}

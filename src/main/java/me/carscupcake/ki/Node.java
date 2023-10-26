package me.carscupcake.ki;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.Getter;
import lombok.Setter;
import me.carscupcake.KiApi;

@Getter
@Setter
public class Node {
    private double bias = 0;
    private WeightedInputs[] connections = new WeightedInputs[]{};
    private double velocity;
    private double costGradiant;
    public JsonNode toJson() {
        ObjectNode jsonObject = KiApi.factory.objectNode();
        jsonObject.put("bias", bias);
        ArrayNode array = KiApi.factory.arrayNode();
        for (WeightedInputs inputs : connections)
            array.add(inputs.toJson());
        jsonObject.set("connections", array);
        return jsonObject;
    }
    public static Node from(ObjectNode object, Layer prev) {
        Node n = new Node();
        n.bias = object.get("bias").asDouble();
        if (Double.isNaN(n.bias)) n.bias = 0;
        if (prev == null) return n;
        ArrayNode array = object.withArray("connections");
        int i = 0;
        n.connections = new WeightedInputs[array.size()];
        for (JsonNode element : array) {
            n.connections[i] = WeightedInputs.from((ObjectNode) element, prev.getNodes()[i]);
            i++;
        }
        return n;
    }

    public void setBias(double bias) {
        this.bias = bias;
        if (Double.isNaN(bias)) this.bias = 0;
    }
}

package me.carscupcake.ki;

import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.Getter;
import lombok.Setter;
import me.carscupcake.KiApi;

import java.util.Random;
@Getter
public class WeightedInputs{
    @Setter
    private double weight;
    private final Node prev;
    @Setter
    private double velocity;
    @Setter
    private double costGradiant;
    public WeightedInputs(Node next, double weight) {
        this.prev = next;
        this.weight = weight;
    }
    public ObjectNode toJson() {
        ObjectNode obj = KiApi.factory.objectNode();
        obj.put("weight", weight);
        obj.put("velocity", velocity);
        return obj;
    }
    public static WeightedInputs from(ObjectNode obj, Node prev) {
         double weight = obj.get("weight").asDouble();
        WeightedInputs inputs = new WeightedInputs(prev, weight);
        inputs.velocity = obj.get("velocity").asDouble();
        return inputs;
    }
}

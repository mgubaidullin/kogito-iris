package one.entropy.demo;

import java.util.Map;

public class IrisClassificator {

    public static Map<String, Double> classify(double ph, double pw, double sh, double sw){
        return Map.of("setosa", 0.33, "virginica", 0.33, "versicolor", 0.34);

    }
}

package org.example.noai;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class AnomalyDetection {

    public static void main(String[] args) {
        Map<String, List<Double>> data = readData("C:\\Users\\Dilit\\IdeaProjects\\DeepLearning4j\\src\\main\\resources\\timeseriesWithnames.csv");
        detectAnomalies(data);
    }

    // Чтение данных из CSV-файла
    private static Map<String, List<Double>> readData(String filename) {
        Map<String, List<Double>> dataMap = new LinkedHashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\t");
                if (parts.length != 2) continue;
                String name = parts[0].trim();
                double value = Double.parseDouble(parts[1].trim());
                dataMap.computeIfAbsent(name, k -> new ArrayList<>()).add(value);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return dataMap;
    }

    // Поиск аномалий с использованием IQR
    private static void detectAnomalies(Map<String, List<Double>> data) {
        for (Map.Entry<String, List<Double>> entry : data.entrySet()) {
            String name = entry.getKey();
            List<Double> values = entry.getValue();

            if (values.size() < 2) {
                System.out.println("Недостаточно данных для: " + name);
                continue;
            }

            List<Double> sortedValues = new ArrayList<>(values);
            Collections.sort(sortedValues);

            double q1 = getPercentile(sortedValues, 25);
            double q3 = getPercentile(sortedValues, 75);
            double iqr = q3 - q1;
            double lowerBound = q1 - 1.5 * iqr;
            double upperBound = q3 + 1.5 * iqr;

            List<Double> anomalies = new ArrayList<>();
            for (Double value : values) {
                if (value < lowerBound || value > upperBound) {
                    anomalies.add(value);
                }
            }

            System.out.println("Аномалии для " + name + ": " + anomalies);
        }
    }

    // Расчет процентиля
    private static double getPercentile(List<Double> sortedData, double percentile) {
        int n = sortedData.size();
        double index = (percentile / 100.0) * (n - 1);
        int lower = (int) Math.floor(index);
        if (lower >= n - 1) return sortedData.get(n - 1);
        double fraction = index - lower;
        return sortedData.get(lower) + fraction * (sortedData.get(lower + 1) - sortedData.get(lower));
    }
}
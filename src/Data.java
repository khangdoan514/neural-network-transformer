package src;

import java.util.Random;

public class Data {
    public static double[][][] makeXor(int samplesPerClass, double noise, int seed) {
        Random randomGenerator = new Random(seed);
        double[][] centers = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
        double[] centerLabels = {0.0, 1.0, 1.0, 0.0};
        int total = samplesPerClass * 4;
        double[][] inputs = new double[total][2];
        double[][] targets = new double[total][1];
        int write = 0;
        for (int centerIndex = 0; centerIndex < 4; centerIndex++) {
            for (int sample = 0; sample < samplesPerClass; sample++) {
                inputs[write][0] = centers[centerIndex][0] + noise * randomGenerator.nextGaussian();
                inputs[write][1] = centers[centerIndex][1] + noise * randomGenerator.nextGaussian();
                targets[write][0] = centerLabels[centerIndex];
                write += 1;
            }
        }

        return shuffleDense(inputs, targets, randomGenerator);
    }

    public static int[][][] makeCopy(int sampleCount, int sequenceLength, int vocabularySize, int seed) {
        Random randomGenerator = new Random(seed);
        int[][] inputs = new int[sampleCount][sequenceLength];
        int[][] targets = new int[sampleCount][sequenceLength];
        for (int sample = 0; sample < sampleCount; sample++) {
            for (int time = 0; time < sequenceLength; time++) {
                int token = randomGenerator.nextInt(vocabularySize);
                inputs[sample][time] = token;
                targets[sample][time] = token;
            }
        }

        return new int[][][] {inputs, targets};
    }

    public static int[][][] makeShift(int sampleCount, int sequenceLength, int vocabularySize, int seed) {
        Random randomGenerator = new Random(seed);
        int[][] inputs = new int[sampleCount][sequenceLength];
        int[][] targets = new int[sampleCount][sequenceLength];
        for (int sample = 0; sample < sampleCount; sample++) {
            for (int time = 0; time < sequenceLength; time++) {
                inputs[sample][time] = randomGenerator.nextInt(vocabularySize);
            }

            for (int time = 0; time < sequenceLength; time++) {
                targets[sample][time] = time + 1 < sequenceLength ? inputs[sample][time + 1] : inputs[sample][time];
            }
        }

        return new int[][][] {inputs, targets};
    }

    public static int[][][] makeTinyText(int sampleCount, int sequenceLength, int seed) {
        String text = "the cat sat on the mat the dog sat on the log the cat sat";
        int vocabularySize = 27;
        Random randomGenerator = new Random(seed);
        int[] encoded = new int[text.length()];
        for (int index = 0; index < text.length(); index++) {
            char character = text.charAt(index);
            encoded[index] = character == ' ' ? 26 : character - 'a';
        }

        int[][] inputs = new int[sampleCount][sequenceLength];
        int[][] targets = new int[sampleCount][sequenceLength];
        for (int sample = 0; sample < sampleCount; sample++) {
            int start = randomGenerator.nextInt(encoded.length - sequenceLength - 1);
            for (int time = 0; time < sequenceLength; time++) {
                inputs[sample][time] = encoded[start + time];
                targets[sample][time] = encoded[start + time + 1];
            }
        }

        return new int[][][] {inputs, targets, new int[][] {{vocabularySize}}};
    }

    private static double[][][] shuffleDense(double[][] inputs, double[][] targets, Random randomGenerator) {
        int sampleCount = inputs.length;
        for (int index = sampleCount - 1; index > 0; index--) {
            int swap = randomGenerator.nextInt(index + 1);
            double[] inputRow = inputs[index];
            inputs[index] = inputs[swap];
            inputs[swap] = inputRow;
            double[] targetRow = targets[index];
            targets[index] = targets[swap];
            targets[swap] = targetRow;
        }

        return new double[][][] {inputs, targets};
    }
}

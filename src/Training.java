package src;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;

public class Training {
    public static class TrainingHistory {
        public List<Double> loss = new ArrayList<>();
        public List<Double> metric = new ArrayList<>();
    }

    public static double accuracyBinary(Tensor logits, double[][] target) {
        int count = 0;
        int total = logits.data.length;
        for (int index = 0; index < total; index++) {
            double prediction = logits.data[index] >= 0.0 ? 1.0 : 0.0;
            double label = target[index / target[0].length][index % target[0].length];
            if (prediction == label) {
                count += 1;
            }
        }

        return (double) count / total;
    }

    public static double accuracyMulticlass(Tensor logits, int[] target) {
        int rows = logits.shape[0];
        int columns = logits.shape[1];
        int count = 0;
        for (int row = 0; row < rows; row++) {
            int best = 0;
            double bestValue = logits.data[row * columns];
            for (int column = 1; column < columns; column++) {
                double value = logits.data[row * columns + column];
                if (value > bestValue) {
                    bestValue = value;
                    best = column;
                }
            }

            if (best == target[row]) {
                count += 1;
            }
        }

        return (double) count / rows;
    }

    public static double accuracyTokens(Tensor logits, int[][] target) {
        int batch = logits.shape[0];
        int sequence = logits.shape[1];
        int vocabularySize = logits.shape[2];
        int count = 0;
        int total = batch * sequence;
        for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
            for (int time = 0; time < sequence; time++) {
                int offset = (batchIndex * sequence + time) * vocabularySize;
                int best = 0;
                double bestValue = logits.data[offset];
                for (int token = 1; token < vocabularySize; token++) {
                    double value = logits.data[offset + token];
                    if (value > bestValue) {
                        bestValue = value;
                        best = token;
                    }
                }

                if (best == target[batchIndex][time]) {
                    count += 1;
                }
            }
        }

        return (double) count / total;
    }

    public static TrainingHistory train(Module model, Module lossFunction, Optimizer optimizer, double[][] inputs, double[][] targets, int epochs, Integer batchSize, BiFunction<Tensor, double[][], Double> metricFunction, Random randomGenerator, boolean verbose, int logEvery) {
        Random generator = randomGenerator == null ? new Random(0) : randomGenerator;
        int sampleCount = inputs.length;
        int resolvedBatchSize = batchSize == null ? sampleCount : batchSize;
        TrainingHistory history = new TrainingHistory();
        for (int epoch = 1; epoch <= epochs; epoch++) {
            int[] permutation = permutation(sampleCount, generator);
            double[][] shuffledInputs = gatherRows(inputs, permutation);
            double[][] shuffledTargets = gatherRows(targets, permutation);
            List<Double> batchLosses = new ArrayList<>();
            List<Double> batchMetrics = new ArrayList<>();
            for (int start = 0; start < sampleCount; start += resolvedBatchSize) {
                int end = Math.min(start + resolvedBatchSize, sampleCount);
                double[][] batchInputs = sliceRows(shuffledInputs, start, end);
                double[][] batchTargets = sliceRows(shuffledTargets, start, end);
                optimizer.zeroGrad();
                Tensor prediction = model.forward(new Tensor(batchInputs, false));
                Tensor loss = invokeLoss(lossFunction, prediction, batchTargets);
                loss.backward();
                optimizer.step();
                batchLosses.add(loss.data[0]);
                if (metricFunction != null) {
                    batchMetrics.add(metricFunction.apply(prediction.detach(), batchTargets));
                }
            }

            double epochLoss = mean(batchLosses);
            double epochMetric = batchMetrics.isEmpty() ? Double.NaN : mean(batchMetrics);
            history.loss.add(epochLoss);
            history.metric.add(epochMetric);
            if (verbose && (epoch % logEvery == 0 || epoch == 1 || epoch == epochs)) {
                if (metricFunction == null) {
                    System.out.printf("epoch %4d | loss %.6f%n", epoch, epochLoss);
                }

                else {
                    System.out.printf("epoch %4d | loss %.6f | metric %.4f%n", epoch, epochLoss, epochMetric);
                }
            }
        }

        return history;
    }

    public static TrainingHistory trainLanguage(Transformer model, CrossEntropyLoss lossFunction, Optimizer optimizer, int[][] inputs, int[][] targets, int epochs, Integer batchSize, boolean verbose, int logEvery, Random randomGenerator) {
        Random generator = randomGenerator == null ? new Random(0) : randomGenerator;
        int sampleCount = inputs.length;
        int resolvedBatchSize = batchSize == null ? sampleCount : batchSize;
        TrainingHistory history = new TrainingHistory();
        for (int epoch = 1; epoch <= epochs; epoch++) {
            int[] permutation = permutation(sampleCount, generator);
            int[][] shuffledInputs = gatherTokenRows(inputs, permutation);
            int[][] shuffledTargets = gatherTokenRows(targets, permutation);
            List<Double> batchLosses = new ArrayList<>();
            List<Double> batchMetrics = new ArrayList<>();
            for (int start = 0; start < sampleCount; start += resolvedBatchSize) {
                int end = Math.min(start + resolvedBatchSize, sampleCount);
                int[][] batchInputs = sliceTokenRows(shuffledInputs, start, end);
                int[][] batchTargets = sliceTokenRows(shuffledTargets, start, end);
                optimizer.zeroGrad();
                Tensor prediction = model.forward(batchInputs);
                Tensor loss = lossFunction.forward(prediction, batchTargets);
                loss.backward();
                optimizer.step();
                batchLosses.add(loss.data[0]);
                batchMetrics.add(accuracyTokens(prediction.detach(), batchTargets));
            }

            double epochLoss = mean(batchLosses);
            double epochMetric = mean(batchMetrics);
            history.loss.add(epochLoss);
            history.metric.add(epochMetric);
            if (verbose && (epoch % logEvery == 0 || epoch == 1 || epoch == epochs)) {
                System.out.printf("epoch %4d | loss %.6f | metric %.4f%n", epoch, epochLoss, epochMetric);
            }
        }

        return history;
    }

    private static Tensor invokeLoss(Module lossFunction, Tensor prediction, double[][] targets) {
        if (lossFunction instanceof MSELoss mseLoss) {
            return mseLoss.forward(prediction, targets);
        }

        if (lossFunction instanceof BCEWithLogitsLoss bceLoss) {
            return bceLoss.forward(prediction, targets);
        }

        throw new IllegalArgumentException("unsupported loss for dense targets");
    }

    private static int[] permutation(int sampleCount, Random randomGenerator) {
        int[] order = new int[sampleCount];
        for (int index = 0; index < sampleCount; index++) {
            order[index] = index;
        }

        for (int index = sampleCount - 1; index > 0; index--) {
            int swap = randomGenerator.nextInt(index + 1);
            int temporary = order[index];
            order[index] = order[swap];
            order[swap] = temporary;
        }

        return order;
    }

    private static double[][] gatherRows(double[][] values, int[] order) {
        double[][] out = new double[values.length][];
        for (int index = 0; index < order.length; index++) {
            out[index] = values[order[index]];
        }

        return out;
    }

    private static int[][] gatherTokenRows(int[][] values, int[] order) {
        int[][] out = new int[values.length][];
        for (int index = 0; index < order.length; index++) {
            out[index] = values[order[index]];
        }

        return out;
    }

    private static double[][] sliceRows(double[][] values, int start, int end) {
        double[][] out = new double[end - start][];
        System.arraycopy(values, start, out, 0, end - start);

        return out;
    }

    private static int[][] sliceTokenRows(int[][] values, int start, int end) {
        int[][] out = new int[end - start][];
        System.arraycopy(values, start, out, 0, end - start);

        return out;
    }

    private static double mean(List<Double> values) {
        double total = 0.0;
        for (double value : values) {
            total += value;
        }

        return total / values.size();
    }
}

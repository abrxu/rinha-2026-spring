package com.abrxu.fraud_detection_rinha;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;

import java.io.*;
import java.util.Random;
import java.util.zip.GZIPInputStream;

public class IndexBuilder {

    private static final int NUM_VECTORS = 3_000_000;
    private static final int DIMENSIONS = 14;
    private static final int NUM_CENTROIDS = 1024;

    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.err.println("Usage: IndexBuilder <references.json.gz> <index.bin>");
            System.exit(1);
        }

        String inputPath = args[0];
        String outputPath = args[1];

        System.out.println("Loading references from: " + inputPath);
        float[] vectors = new float[NUM_VECTORS * DIMENSIONS];
        byte[] labels = new byte[NUM_VECTORS];
        loadReferences(inputPath, vectors, labels);

        System.out.println("Building k-means index...");
        float[] centroids = selectCentroids(vectors);
        int[] clusterCounts = new int[NUM_CENTROIDS];
        int[] vectorToCluster = assignClusters(vectors, centroids, clusterCounts);
        int[] clusterOffsets = computeOffsets(clusterCounts);
        byte[] reorderedLabels = new byte[NUM_VECTORS];
        byte[] quantizedVectors = reorderAndQuantize(vectors, labels, vectorToCluster, clusterOffsets, clusterCounts, reorderedLabels);

        System.out.println("Writing index to: " + outputPath);
        writeIndex(outputPath, quantizedVectors, reorderedLabels, centroids, clusterOffsets, clusterCounts);

        System.out.println("Done.");
    }

    private static void loadReferences(String path, float[] vectors, byte[] labels) throws IOException {
        JsonFactory jsonFactory = new JsonFactory();
        try (InputStream rawStream = new FileInputStream(path);
             GZIPInputStream gzis = new GZIPInputStream(rawStream);
             JsonParser parser = jsonFactory.createParser(gzis)) {

            JsonToken token = parser.nextToken();
            if (token != JsonToken.START_ARRAY) {
                throw new IllegalStateException("Expected START_ARRAY, got " + token);
            }

            int vectorIndex = 0;
            int labelIndex = 0;

            while (parser.nextToken() != JsonToken.END_ARRAY) {
                if (parser.currentToken() != JsonToken.START_OBJECT) {
                    parser.skipChildren();
                    continue;
                }

                float[] vector = new float[DIMENSIONS];
                byte label = 0;

                while (parser.nextToken() != JsonToken.END_OBJECT) {
                    String fieldName = parser.currentName();
                    if ("vector".equals(fieldName)) {
                        parser.nextToken();
                        for (int i = 0; i < DIMENSIONS; i++) {
                            parser.nextToken();
                            vector[i] = (float) parser.getDoubleValue();
                        }
                        parser.nextToken();
                    } else if ("label".equals(fieldName)) {
                        parser.nextToken();
                        label = "fraud".equals(parser.getText()) ? (byte) 1 : (byte) 0;
                    }
                }

                System.arraycopy(vector, 0, vectors, vectorIndex, DIMENSIONS);
                vectorIndex += DIMENSIONS;
                labels[labelIndex++] = label;

                if (labelIndex % 100_000 == 0) {
                    System.out.println("  loaded " + labelIndex + " references...");
                }
            }
        }
        System.out.println("  total: " + NUM_VECTORS + " vectors");
    }

    private static float[] selectCentroids(float[] vectors) {
        float[] centroids = new float[NUM_CENTROIDS * DIMENSIONS];

        Random random = new Random(42);
        int firstIdx = random.nextInt(NUM_VECTORS);
        System.arraycopy(vectors, firstIdx * DIMENSIONS, centroids, 0, DIMENSIONS);

        double[] minDist = new double[NUM_VECTORS];
        for (int i = 0; i < NUM_VECTORS; i++) {
            minDist[i] = squaredDistance(vectors, i, centroids, 0);
        }

        for (int c = 1; c < NUM_CENTROIDS; c++) {
            double totalWeight = 0;
            for (int i = 0; i < NUM_VECTORS; i++) totalWeight += minDist[i];

            double r = random.nextDouble() * totalWeight;
            double cumulative = 0;
            int selected = 0;
            for (int i = 0; i < NUM_VECTORS; i++) {
                cumulative += minDist[i];
                if (cumulative >= r) {
                    selected = i;
                    break;
                }
            }

            System.arraycopy(vectors, selected * DIMENSIONS, centroids, c * DIMENSIONS, DIMENSIONS);

            for (int i = 0; i < NUM_VECTORS; i++) {
                double dist = squaredDistance(vectors, i, centroids, c);
                if (dist < minDist[i]) {
                    minDist[i] = dist;
                }
            }

            if (c % 100 == 0) {
                System.out.println("  centroids: " + c + "/" + NUM_CENTROIDS);
            }
        }

        return centroids;
    }

    private static int[] assignClusters(float[] vectors, float[] centroids, int[] clusterCounts) {
        int[] vectorToCluster = new int[NUM_VECTORS];

        for (int i = 0; i < NUM_VECTORS; i++) {
            int bestCluster = 0;
            double bestDist = Double.MAX_VALUE;

            for (int c = 0; c < NUM_CENTROIDS; c++) {
                double dist = squaredDistance(vectors, i, centroids, c);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestCluster = c;
                }
            }

            vectorToCluster[i] = bestCluster;
            clusterCounts[bestCluster]++;

            if (i % 500_000 == 0) {
                System.out.println("  assigned " + i + " vectors...");
            }
        }

        return vectorToCluster;
    }

    private static int[] computeOffsets(int[] clusterCounts) {
        int[] clusterOffsets = new int[NUM_CENTROIDS];
        int offset = 0;
        for (int c = 0; c < NUM_CENTROIDS; c++) {
            clusterOffsets[c] = offset;
            offset += clusterCounts[c];
        }
        return clusterOffsets;
    }

    private static byte[] reorderAndQuantize(float[] vectors, byte[] labels, int[] vectorToCluster,
                                              int[] clusterOffsets, int[] clusterCounts, byte[] outLabels) {
        byte[] quantizedVectors = new byte[NUM_VECTORS * DIMENSIONS];
        int[] clusterPos = new int[NUM_CENTROIDS];
        System.arraycopy(clusterOffsets, 0, clusterPos, 0, NUM_CENTROIDS);

        for (int i = 0; i < NUM_VECTORS; i++) {
            int cluster = vectorToCluster[i];
            int pos = clusterPos[cluster]++;
            int srcOffset = i * DIMENSIONS;
            int dstOffset = pos * DIMENSIONS;

            for (int d = 0; d < DIMENSIONS; d++) {
                quantizedVectors[dstOffset + d] = quantize(vectors[srcOffset + d]);
            }
            outLabels[pos] = labels[i];
        }

        return quantizedVectors;
    }

    private static byte quantize(float value) {
        int q = (int) Math.round((value + 1.0f) * 127.5f);
        if (q < 0) q = 0;
        if (q > 255) q = 255;
        return (byte) (q - 128);
    }

    private static double squaredDistance(float[] vectors, int vecIdx, float[] centroids, int centroidIdx) {
        double dist = 0.0;
        int vecOffset = vecIdx * DIMENSIONS;
        int centroidOffset = centroidIdx * DIMENSIONS;
        for (int d = 0; d < DIMENSIONS; d++) {
            float diff = vectors[vecOffset + d] - centroids[centroidOffset + d];
            dist += diff * diff;
        }
        return dist;
    }

    private static void writeIndex(String path, byte[] quantizedVectors, byte[] labels,
                                    float[] centroids, int[] clusterOffsets, int[] clusterCounts) throws IOException {
        try (OutputStream os = new FileOutputStream(path);
             BufferedOutputStream bos = new BufferedOutputStream(os);
             DataOutputStream dos = new DataOutputStream(bos)) {

            dos.writeBytes("RINHA");
            dos.writeByte(1);
            dos.writeInt(NUM_VECTORS);
            dos.writeInt(DIMENSIONS);
            dos.writeInt(NUM_CENTROIDS);

            dos.write(quantizedVectors);
            dos.write(labels);

            for (float f : centroids) dos.writeFloat(f);
            for (int i : clusterOffsets) dos.writeInt(i);
            for (int i : clusterCounts) dos.writeInt(i);

            dos.flush();
        }
    }
}

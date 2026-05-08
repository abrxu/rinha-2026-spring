package com.abrxu.fraud_detection_rinha;

import com.abrxu.fraud_detection_rinha.service.VectorStoreService;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.GZIPInputStream;

public class IndexBuilder {

    private static final int EXPECTED_VECTORS = 3_000_000;
    private static final int DIMENSIONS = VectorStoreService.DIMENSIONS;
    private static final int BLOCK_SIZE = VectorStoreService.BLOCK_SIZE;
    private static final int DEFAULT_CENTROIDS = 4096;
    private static final int DEFAULT_ITERATIONS = 10;
    private static final long DEFAULT_SEED = 0xdead_beef_cafe_babeL;
    private static final int KMEANS_INIT_SAMPLE = 65_536;
    private static final short PAD_VALUE = Short.MAX_VALUE;

    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.err.println("Usage: IndexBuilder <references.json.gz> <index.bin>");
            System.exit(1);
        }

        int centroidsCount = intEnv("IVF_K", DEFAULT_CENTROIDS);
        int iterations = intEnv("IVF_ITERS", DEFAULT_ITERATIONS);
        long seed = longEnv("IVF_SEED", DEFAULT_SEED);

        System.out.println("Loading references from: " + args[0]);
        Dataset input = loadReferences(args[0]);

        System.out.printf("Building k-means index: k=%d, iterations=%d%n", centroidsCount, iterations);
        float[] centroids = kmeans(input.vectors, input.count, centroidsCount, iterations, seed);
        short[] assignments = new short[input.count];
        int[] counts = assignClusters(input.vectors, input.count, centroids, centroidsCount, assignments);

        System.out.println("Building blocked IVF layout...");
        Layout layout = buildLayout(input, centroids, assignments, counts, centroidsCount);

        System.out.println("Writing index to: " + args[1]);
        writeIndex(args[1], layout);
        System.out.println("Done.");
    }

    private static int intEnv(String name, int defaultValue) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            return defaultValue;
        }
        return Integer.parseInt(value);
    }

    private static long longEnv(String name, long defaultValue) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            return defaultValue;
        }
        return Long.parseLong(value);
    }

    private static Dataset loadReferences(String path) throws IOException {
        JsonFactory jsonFactory = new JsonFactory();
        float[] vectors = new float[EXPECTED_VECTORS * DIMENSIONS];
        byte[] labels = new byte[EXPECTED_VECTORS];

        try (InputStream rawStream = new FileInputStream(path);
             GZIPInputStream gzis = new GZIPInputStream(rawStream);
             JsonParser parser = jsonFactory.createParser(gzis)) {

            JsonToken token = parser.nextToken();
            if (token != JsonToken.START_ARRAY) {
                throw new IllegalStateException("Expected START_ARRAY, got " + token);
            }

            int count = 0;
            while (parser.nextToken() != JsonToken.END_ARRAY) {
                if (parser.currentToken() != JsonToken.START_OBJECT) {
                    parser.skipChildren();
                    continue;
                }

                int vectorOffset = count * DIMENSIONS;
                byte label = 0;

                while (parser.nextToken() != JsonToken.END_OBJECT) {
                    String fieldName = parser.currentName();
                    if ("vector".equals(fieldName)) {
                        parser.nextToken();
                        for (int i = 0; i < DIMENSIONS; i++) {
                            parser.nextToken();
                            vectors[vectorOffset + i] = (float) parser.getDoubleValue();
                        }
                        parser.nextToken();
                    } else if ("label".equals(fieldName)) {
                        parser.nextToken();
                        label = "fraud".equals(parser.getText()) ? (byte) 1 : (byte) 0;
                    } else {
                        parser.nextToken();
                        parser.skipChildren();
                    }
                }

                labels[count] = label;
                count++;
                if (count % 100_000 == 0) {
                    System.out.println("  loaded " + count + " references...");
                }
            }

            System.out.println("  total: " + count + " vectors");
            return new Dataset(vectors, labels, count);
        }
    }

    private static float[] kmeans(float[] vectors, int count, int centroidsCount, int iterations, long seed) {
        int[] sample = sampleIndices(count, seed);
        float[] centroids = kmeansPlusPlus(vectors, sample, centroidsCount, seed);
        short[] assignments = new short[sample.length];
        Arrays.fill(assignments, (short) -1);

        for (int iteration = 1; iteration <= iterations; iteration++) {
            int[] counts = new int[centroidsCount];
            int changed = assignSampleClusters(vectors, sample, centroids, centroidsCount, assignments, counts);
            updateSampleCentroids(vectors, sample, assignments, centroids, counts, centroidsCount);

            double changedPct = changed * 100.0 / sample.length;
            System.out.printf("  iter %2d/%d: %.3f%% changed%n", iteration, iterations, changedPct);
            if (changed * 1000L < sample.length) {
                System.out.println("  early stop: under 0.1% assignment changes");
                break;
            }
        }

        return centroids;
    }

    private static int[] sampleIndices(int count, long seed) {
        Random random = new Random(seed);
        int sampleSize = Math.min(count, KMEANS_INIT_SAMPLE);
        int[] sample = new int[sampleSize];
        for (int i = 0; i < sampleSize; i++) {
            sample[i] = random.nextInt(count);
        }
        return sample;
    }

    private static float[] kmeansPlusPlus(float[] vectors, int[] sample, int centroidsCount, long seed) {
        System.out.printf("  kmeans++ init: sample=%d%n", sample.length);
        Random random = new Random(seed ^ 0x9e37_79b9_7f4a_7c15L);
        int sampleSize = sample.length;
        float[] centroids = new float[centroidsCount * DIMENSIONS];
        int first = sample[random.nextInt(sampleSize)];
        System.arraycopy(vectors, first * DIMENSIONS, centroids, 0, DIMENSIONS);

        float[] minDistances = new float[sampleSize];
        Arrays.fill(minDistances, Float.POSITIVE_INFINITY);

        for (int c = 1; c < centroidsCount; c++) {
            int lastCentroid = c - 1;
            double totalWeight = 0.0;
            for (int i = 0; i < sampleSize; i++) {
                float distance = squaredDistance(vectors, sample[i], centroids, lastCentroid);
                if (distance < minDistances[i]) {
                    minDistances[i] = distance;
                }
                totalWeight += minDistances[i];
            }

            double pick = random.nextDouble() * totalWeight;
            double cumulative = 0.0;
            int selected = sample[sampleSize - 1];
            for (int i = 0; i < sampleSize; i++) {
                cumulative += minDistances[i];
                if (cumulative >= pick) {
                    selected = sample[i];
                    break;
                }
            }

            System.arraycopy(vectors, selected * DIMENSIONS, centroids, c * DIMENSIONS, DIMENSIONS);
            if (c % 512 == 0) {
                System.out.printf("    centroids: %d/%d%n", c, centroidsCount);
            }
        }

        return centroids;
    }

    private static int assignSampleClusters(
            float[] vectors,
            int[] sample,
            float[] centroids,
            int centroidsCount,
            short[] assignments,
            int[] counts
    ) {
        int changed = 0;
        for (int i = 0; i < sample.length; i++) {
            short next = (short) nearestCentroid(vectors, sample[i], centroids, centroidsCount);
            if (assignments[i] != next) {
                assignments[i] = next;
                changed++;
            }
            counts[next & 0xFFFF]++;
        }
        return changed;
    }

    private static void updateSampleCentroids(
            float[] vectors,
            int[] sample,
            short[] assignments,
            float[] centroids,
            int[] counts,
            int centroidsCount
    ) {
        double[] sums = new double[centroidsCount * DIMENSIONS];
        for (int i = 0; i < sample.length; i++) {
            int cluster = assignments[i] & 0xFFFF;
            int vectorOffset = sample[i] * DIMENSIONS;
            int sumOffset = cluster * DIMENSIONS;
            for (int d = 0; d < DIMENSIONS; d++) {
                sums[sumOffset + d] += vectors[vectorOffset + d];
            }
        }

        for (int c = 0; c < centroidsCount; c++) {
            if (counts[c] == 0) {
                continue;
            }
            int centroidOffset = c * DIMENSIONS;
            double inv = 1.0 / counts[c];
            for (int d = 0; d < DIMENSIONS; d++) {
                centroids[centroidOffset + d] = (float) (sums[centroidOffset + d] * inv);
            }
        }
    }

    private static int[] assignClusters(float[] vectors, int count, float[] centroids, int centroidsCount, short[] assignments) {
        int[] counts = new int[centroidsCount];
        assignClusters(vectors, count, centroids, centroidsCount, assignments, counts);
        return counts;
    }

    private static int assignClusters(
            float[] vectors,
            int count,
            float[] centroids,
            int centroidsCount,
            short[] assignments,
            int[] counts
    ) {
        int workers = Math.min(Runtime.getRuntime().availableProcessors(), 16);
        int chunk = (count + workers - 1) / workers;
        AtomicInteger changed = new AtomicInteger();
        int[][] localCounts = new int[workers][centroidsCount];
        Thread[] threads = new Thread[workers];

        for (int worker = 0; worker < workers; worker++) {
            int workerIndex = worker;
            int start = worker * chunk;
            int end = Math.min(count, start + chunk);
            threads[worker] = new Thread(() -> {
                int localChanged = 0;
                int[] workerCounts = localCounts[workerIndex];
                for (int i = start; i < end; i++) {
                    short next = (short) nearestCentroid(vectors, i, centroids, centroidsCount);
                    if (assignments[i] != next) {
                        assignments[i] = next;
                        localChanged++;
                    }
                    workerCounts[next & 0xFFFF]++;
                }
                changed.addAndGet(localChanged);
            });
            threads[worker].start();
        }

        for (Thread thread : threads) {
            try {
                thread.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("Interrupted while assigning clusters", e);
            }
        }

        for (int[] local : localCounts) {
            for (int c = 0; c < centroidsCount; c++) {
                counts[c] += local[c];
            }
        }
        return changed.get();
    }

    private static int nearestCentroid(float[] vectors, int vectorIndex, float[] centroids, int centroidsCount) {
        int best = 0;
        float bestDistance = Float.POSITIVE_INFINITY;
        int vectorOffset = vectorIndex * DIMENSIONS;

        for (int c = 0; c < centroidsCount; c++) {
            float distance = 0.0f;
            int centroidOffset = c * DIMENSIONS;
            for (int d = 0; d < DIMENSIONS; d++) {
                float diff = vectors[vectorOffset + d] - centroids[centroidOffset + d];
                distance += diff * diff;
            }
            if (distance < bestDistance) {
                bestDistance = distance;
                best = c;
            }
        }
        return best;
    }

    private static float squaredDistance(float[] vectors, int vectorIndex, float[] centroids, int centroidIndex) {
        float distance = 0.0f;
        int vectorOffset = vectorIndex * DIMENSIONS;
        int centroidOffset = centroidIndex * DIMENSIONS;
        for (int d = 0; d < DIMENSIONS; d++) {
            float diff = vectors[vectorOffset + d] - centroids[centroidOffset + d];
            distance += diff * diff;
        }
        return distance;
    }

    private static Layout buildLayout(Dataset input, float[] centroids, short[] assignments, int[] counts, int centroidsCount) {
        int[] offsets = new int[centroidsCount + 1];
        int[] blockOffsets = new int[centroidsCount + 1];
        for (int c = 0; c < centroidsCount; c++) {
            offsets[c + 1] = offsets[c] + counts[c];
            blockOffsets[c + 1] = blockOffsets[c] + (counts[c] + BLOCK_SIZE - 1) / BLOCK_SIZE;
        }

        int totalBlocks = blockOffsets[centroidsCount];
        short[] blocks = new short[totalBlocks * DIMENSIONS * BLOCK_SIZE];
        Arrays.fill(blocks, PAD_VALUE);
        byte[] labels = new byte[input.count];
        int[] originalIds = new int[input.count];
        short[] bboxMin = new short[centroidsCount * DIMENSIONS];
        short[] bboxMax = new short[centroidsCount * DIMENSIONS];
        Arrays.fill(bboxMin, Short.MAX_VALUE);
        Arrays.fill(bboxMax, Short.MIN_VALUE);

        int[] writePositions = Arrays.copyOf(offsets, centroidsCount);
        for (int i = 0; i < input.count; i++) {
            int cluster = assignments[i] & 0xFFFF;
            int position = writePositions[cluster]++;
            labels[position] = input.labels[i];
            originalIds[position] = i;

            int positionInCluster = position - offsets[cluster];
            int blockLocal = positionInCluster / BLOCK_SIZE;
            int lane = positionInCluster % BLOCK_SIZE;
            int block = blockOffsets[cluster] + blockLocal;
            int blockBase = block * DIMENSIONS * BLOCK_SIZE;
            int vectorOffset = i * DIMENSIONS;
            int bboxBase = cluster * DIMENSIONS;

            for (int d = 0; d < DIMENSIONS; d++) {
                short quantized = VectorStoreService.quantizeFixed(input.vectors[vectorOffset + d]);
                blocks[blockBase + d * BLOCK_SIZE + lane] = quantized;
                if (quantized < bboxMin[bboxBase + d]) {
                    bboxMin[bboxBase + d] = quantized;
                }
                if (quantized > bboxMax[bboxBase + d]) {
                    bboxMax[bboxBase + d] = quantized;
                }
            }
        }

        for (int c = 0; c < centroidsCount; c++) {
            if (counts[c] == 0) {
                int base = c * DIMENSIONS;
                for (int d = 0; d < DIMENSIONS; d++) {
                    bboxMin[base + d] = 0;
                    bboxMax[base + d] = 0;
                }
            }
        }

        float[] centroidsSoa = new float[centroidsCount * DIMENSIONS];
        for (int c = 0; c < centroidsCount; c++) {
            for (int d = 0; d < DIMENSIONS; d++) {
                centroidsSoa[d * centroidsCount + c] = centroids[c * DIMENSIONS + d];
            }
        }

        return new Layout(
                input.count,
                centroidsCount,
                centroidsSoa,
                bboxMin,
                bboxMax,
                offsets,
                blockOffsets,
                blocks,
                labels,
                originalIds
        );
    }

    private static void writeIndex(String path, Layout layout) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path), 1 << 20))) {
            dos.writeBytes("RIVF");
            dos.writeInt(2);
            dos.writeInt(layout.numVectors);
            dos.writeInt(layout.numCentroids);
            dos.writeInt(DIMENSIONS);
            dos.writeFloat(VectorStoreService.FIXED_SCALE);

            for (float value : layout.centroidsSoa) {
                dos.writeFloat(value);
            }
            for (short value : layout.bboxMin) {
                dos.writeShort(value);
            }
            for (short value : layout.bboxMax) {
                dos.writeShort(value);
            }
            for (int value : layout.offsets) {
                dos.writeInt(value);
            }
            for (int value : layout.blockOffsets) {
                dos.writeInt(value);
            }
            for (short value : layout.blocks) {
                dos.writeShort(value);
            }
            dos.write(layout.labels);
            for (int value : layout.originalIds) {
                dos.writeInt(value);
            }
        }
    }

    private record Dataset(float[] vectors, byte[] labels, int count) {
    }

    private record Layout(
            int numVectors,
            int numCentroids,
            float[] centroidsSoa,
            short[] bboxMin,
            short[] bboxMax,
            int[] offsets,
            int[] blockOffsets,
            short[] blocks,
            byte[] labels,
            int[] originalIds
    ) {
    }
}

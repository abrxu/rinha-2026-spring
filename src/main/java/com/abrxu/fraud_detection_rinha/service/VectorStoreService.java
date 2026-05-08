package com.abrxu.fraud_detection_rinha.service;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.type.MapType;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.util.zip.GZIPInputStream;

@Service
public class VectorStoreService {

    private static final int NUM_VECTORS = 3_000_000;
    private static final int DIMENSIONS = 14;
    private static final int K = 5;
    private static final int NUM_CENTROIDS = 1024;
    private static final int NUM_CLUSTERS_TO_SEARCH = 10;

    private byte[] quantizedVectors;
    private byte[] labels;
    private float[] centroids;
    private int[] clusterOffsets;
    private int[] clusterCounts;
    private int[] vectorToCluster;

    private Map<String, Float> mccRisk;
    private Map<String, Float> normalizationConstants;

    private final JsonFactory jsonFactory = new JsonFactory();
    private final ObjectMapper objectMapper = new ObjectMapper();

    @PostConstruct
    public void init() throws IOException {
        loadNormalizationConstants();
        loadMccRisk();
        loadReferences();
    }

    private void loadNormalizationConstants() throws IOException {
        try (InputStream is = getClass().getClassLoader().getResourceAsStream("normalization.json")) {
            MapType mapType = objectMapper.getTypeFactory().constructMapType(Map.class, String.class, Double.class);
            Map<String, Double> raw = objectMapper.readValue(is, mapType);
            normalizationConstants = new java.util.HashMap<>();
            for (Map.Entry<String, Double> e : raw.entrySet()) {
                normalizationConstants.put(e.getKey(), e.getValue().floatValue());
            }
        }
    }

    private void loadMccRisk() throws IOException {
        try (InputStream is = getClass().getClassLoader().getResourceAsStream("mcc_risk.json")) {
            MapType mapType = objectMapper.getTypeFactory().constructMapType(Map.class, String.class, Double.class);
            Map<String, Double> raw = objectMapper.readValue(is, mapType);
            mccRisk = new java.util.HashMap<>();
            for (Map.Entry<String, Double> e : raw.entrySet()) {
                mccRisk.put(e.getKey(), e.getValue().floatValue());
            }
        }
    }

    private void loadReferences() throws IOException {
        float[] vectors = new float[NUM_VECTORS * DIMENSIONS];
        labels = new byte[NUM_VECTORS];

        InputStream rawStream = getClass().getClassLoader().getResourceAsStream("references.json.gz");
        if (rawStream == null) {
            throw new IllegalStateException("references.json.gz not found in classpath");
        }

        try (GZIPInputStream gzis = new GZIPInputStream(rawStream);
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
            }
        }

        buildIndex(vectors);
    }

    private void buildIndex(float[] vectors) {
        selectCentroids(vectors);
        assignClusters(vectors);
        reorderVectors(vectors);
    }

    private void selectCentroids(float[] vectors) {
        centroids = new float[NUM_CENTROIDS * DIMENSIONS];

        java.util.Random random = new java.util.Random(42);
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
        }
    }

    private void assignClusters(float[] vectors) {
        vectorToCluster = new int[NUM_VECTORS];
        clusterCounts = new int[NUM_CENTROIDS];

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
        }

        clusterOffsets = new int[NUM_CENTROIDS];
        int offset = 0;
        for (int c = 0; c < NUM_CENTROIDS; c++) {
            clusterOffsets[c] = offset;
            offset += clusterCounts[c];
        }
    }

    private void reorderVectors(float[] vectors) {
        quantizedVectors = new byte[NUM_VECTORS * DIMENSIONS];
        int[] clusterPos = new int[NUM_CENTROIDS];
        System.arraycopy(clusterOffsets, 0, clusterPos, 0, NUM_CENTROIDS);

        byte[] newLabels = new byte[NUM_VECTORS];

        for (int i = 0; i < NUM_VECTORS; i++) {
            int cluster = vectorToCluster[i];
            int pos = clusterPos[cluster]++;
            int srcOffset = i * DIMENSIONS;
            int dstOffset = pos * DIMENSIONS;

            for (int d = 0; d < DIMENSIONS; d++) {
                quantizedVectors[dstOffset + d] = quantize(vectors[srcOffset + d]);
            }
            newLabels[pos] = labels[i];
        }

        System.arraycopy(newLabels, 0, labels, 0, NUM_VECTORS);
    }

    private static byte quantize(float value) {
        int q = (int) Math.round((value + 1.0f) * 127.5f);
        if (q < 0) q = 0;
        if (q > 255) q = 255;
        return (byte) (q - 128);
    }

    private static int dequantize(byte q) {
        return q & 0xFF;
    }

    public float getNormalizationConstant(String key) {
        Float value = normalizationConstants.get(key);
        if (value == null) {
            throw new IllegalArgumentException("Unknown normalization constant: " + key);
        }
        return value;
    }

    public float getMccRisk(String mcc) {
        Float risk = mccRisk.get(mcc);
        return risk != null ? risk : 0.5f;
    }

    public double computeFraudScore(float[] queryVector) {
        byte[] queryQuantized = new byte[DIMENSIONS];
        float[] queryFloat = new float[DIMENSIONS];
        for (int d = 0; d < DIMENSIONS; d++) {
            queryQuantized[d] = quantize(queryVector[d]);
            queryFloat[d] = queryVector[d];
        }

        int[] topClusters = new int[NUM_CLUSTERS_TO_SEARCH];
        double[] topClusterDists = new double[NUM_CLUSTERS_TO_SEARCH];
        for (int i = 0; i < NUM_CLUSTERS_TO_SEARCH; i++) {
            topClusterDists[i] = Double.MAX_VALUE;
        }

        for (int c = 0; c < NUM_CENTROIDS; c++) {
            double dist = 0.0;
            int centroidOffset = c * DIMENSIONS;
            for (int d = 0; d < DIMENSIONS; d++) {
                float diff = centroids[centroidOffset + d] - queryFloat[d];
                dist += diff * diff;
            }
            if (dist < topClusterDists[NUM_CLUSTERS_TO_SEARCH - 1]) {
                int insertPos = NUM_CLUSTERS_TO_SEARCH - 1;
                while (insertPos > 0 && dist < topClusterDists[insertPos - 1]) {
                    insertPos--;
                }
                for (int j = NUM_CLUSTERS_TO_SEARCH - 1; j > insertPos; j--) {
                    topClusterDists[j] = topClusterDists[j - 1];
                    topClusters[j] = topClusters[j - 1];
                }
                topClusterDists[insertPos] = dist;
                topClusters[insertPos] = c;
            }
        }

        int[] topIndices = new int[K];
        double[] topDistances = new double[K];
        for (int i = 0; i < K; i++) {
            topDistances[i] = Double.MAX_VALUE;
        }

        for (int ci = 0; ci < NUM_CLUSTERS_TO_SEARCH; ci++) {
            int cluster = topClusters[ci];
            int clusterStart = clusterOffsets[cluster];
            int clusterEnd = clusterStart + clusterCounts[cluster];

            for (int i = clusterStart; i < clusterEnd; i++) {
                int dist = 0;
                int offset = i * DIMENSIONS;
                for (int d = 0; d < DIMENSIONS; d++) {
                    int diff = (quantizedVectors[offset + d] & 0xFF) - (queryQuantized[d] & 0xFF);
                    dist += diff * diff;
                }

                if (dist < topDistances[K - 1]) {
                    int insertPos = K - 1;
                    while (insertPos > 0 && dist < topDistances[insertPos - 1]) {
                        insertPos--;
                    }

                    for (int j = K - 1; j > insertPos; j--) {
                        topDistances[j] = topDistances[j - 1];
                        topIndices[j] = topIndices[j - 1];
                    }
                    topDistances[insertPos] = dist;
                    topIndices[insertPos] = i;
                }
            }
        }

        int fraudCount = 0;
        for (int i = 0; i < K; i++) {
            fraudCount += labels[topIndices[i]];
        }

        return (double) fraudCount / K;
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

}

package com.abrxu.fraud_detection_rinha.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.type.MapType;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import java.io.*;
import java.util.Map;

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

    private Map<String, Float> mccRisk;
    private Map<String, Float> normalizationConstants;

    private final ObjectMapper objectMapper = new ObjectMapper();

    @PostConstruct
    public void init() throws IOException {
        loadNormalizationConstants();
        loadMccRisk();
        loadIndex();
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

    private void loadIndex() throws IOException {
        try (InputStream is = getClass().getClassLoader().getResourceAsStream("index.bin");
             BufferedInputStream bis = new BufferedInputStream(is);
             DataInputStream dis = new DataInputStream(bis)) {

            byte[] magic = new byte[5];
            dis.readFully(magic);
            if (magic[0] != 'R' || magic[1] != 'I' || magic[2] != 'N' || magic[3] != 'H' || magic[4] != 'A') {
                throw new IllegalStateException("Invalid index file");
            }

            int version = dis.readByte();
            if (version != 1) {
                throw new IllegalStateException("Unsupported index version: " + version);
            }

            int nv = dis.readInt();
            int dims = dis.readInt();
            int nc = dis.readInt();

            quantizedVectors = new byte[nv * dims];
            dis.readFully(quantizedVectors);

            labels = new byte[nv];
            dis.readFully(labels);

            centroids = new float[nc * dims];
            for (int i = 0; i < centroids.length; i++) {
                centroids[i] = dis.readFloat();
            }

            clusterOffsets = new int[nc];
            for (int i = 0; i < nc; i++) {
                clusterOffsets[i] = dis.readInt();
            }

            clusterCounts = new int[nc];
            for (int i = 0; i < nc; i++) {
                clusterCounts[i] = dis.readInt();
            }
        }
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

    private static byte quantize(float value) {
        int q = (int) Math.round((value + 1.0f) * 127.5f);
        if (q < 0) q = 0;
        if (q > 255) q = 255;
        return (byte) (q - 128);
    }
}

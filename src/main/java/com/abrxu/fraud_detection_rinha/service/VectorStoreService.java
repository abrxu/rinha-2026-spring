package com.abrxu.fraud_detection_rinha.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.type.MapType;
import jakarta.annotation.PostConstruct;
import org.springframework.stereotype.Service;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

@Service
public class VectorStoreService {

    public static final int DIMENSIONS = 14;
    public static final int K_NEIGHBORS = 5;
    public static final int BLOCK_SIZE = 8;
    public static final float FIXED_SCALE = 10_000.0f;

    private static final int FAST_NPROBE = 8;
    private static final int INDEX_VERSION = 2;

    private int numVectors;
    private int numCentroids;
    private float[] centroidsSoa;
    private short[] bboxMin;
    private short[] bboxMax;
    private int[] offsets;
    private int[] blockOffsets;
    private short[] blocks;
    private byte[] labels;
    private int[] originalIds;

    private Map<String, Float> mccRisk;
    private Map<String, Float> normalizationConstants;

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final ThreadLocal<SearchScratch> scratch = ThreadLocal.withInitial(SearchScratch::new);

    public VectorStoreService() {
    }

    VectorStoreService(
            float[] centroidsSoa,
            short[] bboxMin,
            short[] bboxMax,
            int[] offsets,
            int[] blockOffsets,
            short[] blocks,
            byte[] labels,
            int[] originalIds
    ) {
        this.numCentroids = offsets.length - 1;
        this.numVectors = offsets[this.numCentroids];
        this.centroidsSoa = centroidsSoa;
        this.bboxMin = bboxMin;
        this.bboxMax = bboxMax;
        this.offsets = offsets;
        this.blockOffsets = blockOffsets;
        this.blocks = blocks;
        this.labels = labels;
        this.originalIds = originalIds;
    }

    @PostConstruct
    public void init() throws IOException {
        loadNormalizationConstants();
        loadMccRisk();
        loadIndex();
    }

    private void loadNormalizationConstants() throws IOException {
        try (InputStream is = requireResource("normalization.json")) {
            MapType mapType = objectMapper.getTypeFactory().constructMapType(Map.class, String.class, Double.class);
            Map<String, Double> raw = objectMapper.readValue(is, mapType);
            normalizationConstants = new HashMap<>();
            for (Map.Entry<String, Double> e : raw.entrySet()) {
                normalizationConstants.put(e.getKey(), e.getValue().floatValue());
            }
        }
    }

    private void loadMccRisk() throws IOException {
        try (InputStream is = requireResource("mcc_risk.json")) {
            MapType mapType = objectMapper.getTypeFactory().constructMapType(Map.class, String.class, Double.class);
            Map<String, Double> raw = objectMapper.readValue(is, mapType);
            mccRisk = new HashMap<>();
            for (Map.Entry<String, Double> e : raw.entrySet()) {
                mccRisk.put(e.getKey(), e.getValue().floatValue());
            }
        }
    }

    private InputStream requireResource(String name) throws IOException {
        InputStream is = getClass().getClassLoader().getResourceAsStream(name);
        if (is == null) {
            throw new IOException("Missing resource: " + name);
        }
        return is;
    }

    private void loadIndex() throws IOException {
        try (InputStream is = requireResource("index.bin");
             BufferedInputStream bis = new BufferedInputStream(is, 1 << 20);
             DataInputStream dis = new DataInputStream(bis)) {

            byte[] magic = new byte[4];
            dis.readFully(magic);
            if (magic[0] != 'R' || magic[1] != 'I' || magic[2] != 'V' || magic[3] != 'F') {
                throw new IllegalStateException("Invalid index file");
            }

            int version = dis.readInt();
            if (version != INDEX_VERSION) {
                throw new IllegalStateException("Unsupported index version: " + version);
            }

            numVectors = dis.readInt();
            numCentroids = dis.readInt();
            int dims = dis.readInt();
            float scale = dis.readFloat();
            if (dims != DIMENSIONS) {
                throw new IllegalStateException("Unexpected dimensions: " + dims);
            }
            if (scale != FIXED_SCALE) {
                throw new IllegalStateException("Unexpected quantization scale: " + scale);
            }

            centroidsSoa = new float[numCentroids * DIMENSIONS];
            for (int i = 0; i < centroidsSoa.length; i++) {
                centroidsSoa[i] = dis.readFloat();
            }

            bboxMin = new short[numCentroids * DIMENSIONS];
            for (int i = 0; i < bboxMin.length; i++) {
                bboxMin[i] = dis.readShort();
            }

            bboxMax = new short[numCentroids * DIMENSIONS];
            for (int i = 0; i < bboxMax.length; i++) {
                bboxMax[i] = dis.readShort();
            }

            offsets = new int[numCentroids + 1];
            for (int i = 0; i < offsets.length; i++) {
                offsets[i] = dis.readInt();
            }

            blockOffsets = new int[numCentroids + 1];
            for (int i = 0; i < blockOffsets.length; i++) {
                blockOffsets[i] = dis.readInt();
            }

            int totalBlocks = blockOffsets[numCentroids];
            blocks = new short[totalBlocks * DIMENSIONS * BLOCK_SIZE];
            for (int i = 0; i < blocks.length; i++) {
                blocks[i] = dis.readShort();
            }

            labels = new byte[numVectors];
            dis.readFully(labels);

            originalIds = new int[numVectors];
            for (int i = 0; i < originalIds.length; i++) {
                originalIds[i] = dis.readInt();
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
        return computeFraudCount(queryVector) / (double) K_NEIGHBORS;
    }

    int computeFraudCount(float[] queryVector) {
        SearchScratch s = scratch.get();
        s.ensureCentroids(numCentroids);
        s.reset();

        for (int d = 0; d < DIMENSIONS; d++) {
            s.query[d] = quantizeFixed(queryVector[d]);
        }

        selectFastClusters(queryVector, s);
        for (int i = 0; i < s.fastCount; i++) {
            int cluster = s.topClusters[i];
            s.scanned[cluster] = true;
            s.scannedClusters[s.scannedCount++] = cluster;
            scanCluster(cluster, s);
        }

        int fraudCount = s.fraudCount();
        if (fraudCount != 2 && fraudCount != 3) {
            return fraudCount;
        }

        long worst = s.topDistances[s.worstIndex];
        for (int c = 0; c < numCentroids; c++) {
            if (s.scanned[c]) {
                continue;
            }
            if (bboxLowerBound(s.query, c) <= worst) {
                scanCluster(c, s);
                worst = s.topDistances[s.worstIndex];
            }
        }

        return s.fraudCount();
    }

    private void selectFastClusters(float[] queryVector, SearchScratch s) {
        s.fastCount = Math.min(FAST_NPROBE, numCentroids);
        Arrays.fill(s.topClusterDistances, Float.POSITIVE_INFINITY);

        for (int c = 0; c < numCentroids; c++) {
            float dist = 0.0f;
            for (int d = 0; d < DIMENSIONS; d++) {
                float diff = centroidsSoa[d * numCentroids + c] - queryVector[d];
                dist += diff * diff;
            }

            if (dist < s.topClusterDistances[s.fastCount - 1]) {
                int pos = s.fastCount - 1;
                while (pos > 0 && dist < s.topClusterDistances[pos - 1]) {
                    pos--;
                }
                for (int j = s.fastCount - 1; j > pos; j--) {
                    s.topClusterDistances[j] = s.topClusterDistances[j - 1];
                    s.topClusters[j] = s.topClusters[j - 1];
                }
                s.topClusterDistances[pos] = dist;
                s.topClusters[pos] = c;
            }
        }
    }

    private long bboxLowerBound(short[] query, int cluster) {
        long distance = 0L;
        int base = cluster * DIMENSIONS;
        for (int d = 0; d < DIMENSIONS; d++) {
            int q = query[d];
            int min = bboxMin[base + d];
            int max = bboxMax[base + d];
            int diff;
            if (q < min) {
                diff = min - q;
            } else if (q > max) {
                diff = q - max;
            } else {
                diff = 0;
            }
            distance += (long) diff * diff;
        }
        return distance;
    }

    private void scanCluster(int cluster, SearchScratch s) {
        int start = offsets[cluster];
        int end = offsets[cluster + 1];
        if (start >= end) {
            return;
        }

        int blockStart = blockOffsets[cluster];
        int blockEnd = blockOffsets[cluster + 1];
        int clusterSize = end - start;

        for (int blockLocal = 0, block = blockStart; block < blockEnd; blockLocal++, block++) {
            int lanes = Math.min(BLOCK_SIZE, clusterSize - blockLocal * BLOCK_SIZE);
            int blockBase = block * DIMENSIONS * BLOCK_SIZE;

            for (int lane = 0; lane < lanes; lane++) {
                long distance = 0L;
                long worst = s.topDistances[s.worstIndex];
                for (int d = 0; d < DIMENSIONS; d++) {
                    int diff = blocks[blockBase + d * BLOCK_SIZE + lane] - s.query[d];
                    distance += (long) diff * diff;
                    if (distance > worst) {
                        break;
                    }
                }

                int global = start + blockLocal * BLOCK_SIZE + lane;
                s.tryInsert(distance, labels[global], originalIds[global]);
            }
        }
    }

    public static short quantizeFixed(float value) {
        float clamped = Math.max(-1.0f, Math.min(1.0f, value));
        float scaled = clamped * FIXED_SCALE;
        int rounded = scaled >= 0.0f ? (int) (scaled + 0.5f) : (int) (scaled - 0.5f);
        return (short) rounded;
    }

    private static final class SearchScratch {
        final short[] query = new short[DIMENSIONS];
        final int[] topClusters = new int[FAST_NPROBE];
        final float[] topClusterDistances = new float[FAST_NPROBE];
        final long[] topDistances = new long[K_NEIGHBORS];
        final byte[] topLabels = new byte[K_NEIGHBORS];
        final int[] topOriginalIds = new int[K_NEIGHBORS];

        boolean[] scanned = new boolean[0];
        int[] scannedClusters = new int[0];
        int scannedCount;
        int fastCount;
        int worstIndex;

        void ensureCentroids(int numCentroids) {
            if (scanned.length < numCentroids) {
                scanned = new boolean[numCentroids];
                scannedClusters = new int[numCentroids];
            }
        }

        void reset() {
            for (int i = 0; i < scannedCount; i++) {
                scanned[scannedClusters[i]] = false;
            }
            scannedCount = 0;
            worstIndex = 0;
            Arrays.fill(topDistances, Long.MAX_VALUE);
            Arrays.fill(topLabels, (byte) 0);
            Arrays.fill(topOriginalIds, Integer.MAX_VALUE);
        }

        void tryInsert(long distance, byte label, int originalId) {
            long worst = topDistances[worstIndex];
            int worstOriginalId = topOriginalIds[worstIndex];
            if (distance > worst || (distance == worst && originalId >= worstOriginalId)) {
                return;
            }

            topDistances[worstIndex] = distance;
            topLabels[worstIndex] = label;
            topOriginalIds[worstIndex] = originalId;

            int nextWorst = 0;
            for (int i = 1; i < K_NEIGHBORS; i++) {
                if (topDistances[i] > topDistances[nextWorst]
                        || (topDistances[i] == topDistances[nextWorst]
                        && topOriginalIds[i] > topOriginalIds[nextWorst])) {
                    nextWorst = i;
                }
            }
            worstIndex = nextWorst;
        }

        int fraudCount() {
            int count = 0;
            for (byte label : topLabels) {
                count += label;
            }
            return count;
        }
    }
}

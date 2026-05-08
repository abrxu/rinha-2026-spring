package com.abrxu.fraud_detection_rinha.service;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class VectorStoreServiceTest {

    @Test
    void quantizeFixed_preservesSentinelAndUnitRange() {
        assertEquals(-10_000, VectorStoreService.quantizeFixed(-1.0f));
        assertEquals(0, VectorStoreService.quantizeFixed(0.0f));
        assertEquals(5_000, VectorStoreService.quantizeFixed(0.5f));
        assertEquals(10_000, VectorStoreService.quantizeFixed(1.0f));
        assertEquals(10_000, VectorStoreService.quantizeFixed(2.0f));
        assertEquals(-10_000, VectorStoreService.quantizeFixed(-2.0f));
    }

    @Test
    void quantizeFixed_keepsMissingSentinelFartherFromZeroThanItself() {
        short missing = VectorStoreService.quantizeFixed(-1.0f);
        short zero = VectorStoreService.quantizeFixed(0.0f);

        assertTrue(squaredDiff(missing, missing) < squaredDiff(missing, zero));
    }

    @Test
    void computeFraudCount_usesTopFiveAndOriginalIdTieBreak() {
        VectorStoreService store = syntheticStore(
                new float[][]{{0.0f}},
                new Entry[]{
                        new Entry(0, 0.0f, (byte) 1, 0),
                        new Entry(0, 0.0f, (byte) 1, 1),
                        new Entry(0, 0.0f, (byte) 1, 2),
                        new Entry(0, 0.0f, (byte) 1, 3),
                        new Entry(0, 0.0f, (byte) 1, 4),
                        new Entry(0, 0.0f, (byte) 0, 5),
                }
        );

        assertEquals(5, store.computeFraudCount(query(0.0f)));
    }

    @Test
    void computeFraudCount_widensSearchWhenFastVoteIsBorderline() {
        float[][] centroids = new float[9][1];
        centroids[8][0] = 1.0f;

        VectorStoreService store = syntheticStore(
                centroids,
                new Entry[]{
                        new Entry(0, 0.1f, (byte) 1, 0),
                        new Entry(1, 0.1f, (byte) 1, 1),
                        new Entry(2, 0.1f, (byte) 0, 2),
                        new Entry(3, 0.1f, (byte) 0, 3),
                        new Entry(4, 0.1f, (byte) 0, 4),
                        new Entry(5, 0.1f, (byte) 0, 5),
                        new Entry(6, 0.1f, (byte) 0, 6),
                        new Entry(7, 0.1f, (byte) 0, 7),
                        new Entry(8, 0.0f, (byte) 1, 8),
                }
        );

        assertEquals(3, store.computeFraudCount(query(0.0f)));
    }

    private static long squaredDiff(short left, short right) {
        int diff = left - right;
        return (long) diff * diff;
    }

    private static float[] query(float firstDimension) {
        float[] query = new float[VectorStoreService.DIMENSIONS];
        query[0] = firstDimension;
        return query;
    }

    private static VectorStoreService syntheticStore(float[][] centroidFirstDimensions, Entry[] entries) {
        int centroidsCount = centroidFirstDimensions.length;
        int[] counts = new int[centroidsCount];
        for (Entry entry : entries) {
            counts[entry.cluster()]++;
        }

        int[] offsets = new int[centroidsCount + 1];
        int[] blockOffsets = new int[centroidsCount + 1];
        for (int c = 0; c < centroidsCount; c++) {
            offsets[c + 1] = offsets[c] + counts[c];
            blockOffsets[c + 1] = blockOffsets[c] + (counts[c] + VectorStoreService.BLOCK_SIZE - 1) / VectorStoreService.BLOCK_SIZE;
        }

        int totalBlocks = blockOffsets[centroidsCount];
        short[] blocks = new short[totalBlocks * VectorStoreService.DIMENSIONS * VectorStoreService.BLOCK_SIZE];
        Arrays.fill(blocks, Short.MAX_VALUE);
        byte[] labels = new byte[entries.length];
        int[] originalIds = new int[entries.length];
        short[] bboxMin = new short[centroidsCount * VectorStoreService.DIMENSIONS];
        short[] bboxMax = new short[centroidsCount * VectorStoreService.DIMENSIONS];
        Arrays.fill(bboxMin, Short.MAX_VALUE);
        Arrays.fill(bboxMax, Short.MIN_VALUE);

        int[] writePositions = Arrays.copyOf(offsets, centroidsCount);
        for (Entry entry : entries) {
            int position = writePositions[entry.cluster()]++;
            labels[position] = entry.label();
            originalIds[position] = entry.originalId();

            int positionInCluster = position - offsets[entry.cluster()];
            int block = blockOffsets[entry.cluster()] + positionInCluster / VectorStoreService.BLOCK_SIZE;
            int lane = positionInCluster % VectorStoreService.BLOCK_SIZE;
            int blockBase = block * VectorStoreService.DIMENSIONS * VectorStoreService.BLOCK_SIZE;
            int bboxBase = entry.cluster() * VectorStoreService.DIMENSIONS;

            for (int d = 0; d < VectorStoreService.DIMENSIONS; d++) {
                float value = d == 0 ? entry.firstDimension() : 0.0f;
                short quantized = VectorStoreService.quantizeFixed(value);
                blocks[blockBase + d * VectorStoreService.BLOCK_SIZE + lane] = quantized;
                bboxMin[bboxBase + d] = (short) Math.min(bboxMin[bboxBase + d], quantized);
                bboxMax[bboxBase + d] = (short) Math.max(bboxMax[bboxBase + d], quantized);
            }
        }

        for (int c = 0; c < centroidsCount; c++) {
            if (counts[c] == 0) {
                int base = c * VectorStoreService.DIMENSIONS;
                for (int d = 0; d < VectorStoreService.DIMENSIONS; d++) {
                    bboxMin[base + d] = 0;
                    bboxMax[base + d] = 0;
                }
            }
        }

        float[] centroidsSoa = new float[centroidsCount * VectorStoreService.DIMENSIONS];
        for (int c = 0; c < centroidsCount; c++) {
            centroidsSoa[c] = centroidFirstDimensions[c][0];
        }

        return new VectorStoreService(centroidsSoa, bboxMin, bboxMax, offsets, blockOffsets, blocks, labels, originalIds);
    }

    private record Entry(int cluster, float firstDimension, byte label, int originalId) {
    }
}

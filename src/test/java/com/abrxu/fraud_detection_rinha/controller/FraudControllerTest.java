package com.abrxu.fraud_detection_rinha.controller;

import com.abrxu.fraud_detection_rinha.service.VectorStoreService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.time.LocalDateTime;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class FraudControllerTest {

    private VectorStoreService vectorStore;
    private FraudController controller;

    @BeforeEach
    void setUp() {
        vectorStore = mock(VectorStoreService.class);
        when(vectorStore.getNormalizationConstant("max_amount")).thenReturn(10000f);
        when(vectorStore.getNormalizationConstant("max_installments")).thenReturn(12f);
        when(vectorStore.getNormalizationConstant("amount_vs_avg_ratio")).thenReturn(10f);
        when(vectorStore.getNormalizationConstant("max_minutes")).thenReturn(1440f);
        when(vectorStore.getNormalizationConstant("max_km")).thenReturn(1000f);
        when(vectorStore.getNormalizationConstant("max_tx_count_24h")).thenReturn(20f);
        when(vectorStore.getNormalizationConstant("max_merchant_avg_amount")).thenReturn(10000f);
        when(vectorStore.getMccRisk("5411")).thenReturn(0.15f);
        when(vectorStore.getMccRisk("7802")).thenReturn(0.75f);

        controller = new FraudController(vectorStore);
    }

    @Test
    void normalize_legitTransaction_noLastTransaction() {
        FraudRequest request = new FraudRequest(
                "tx-1329056812",
                new FraudRequest.Transaction(41.12, 2, LocalDateTime.of(2026, 3, 11, 18, 45, 53)),
                new FraudRequest.Customer(82.24, 3, List.of("MERC-003", "MERC-016")),
                new FraudRequest.Merchant("MERC-016", "5411", 60.25),
                new FraudRequest.Terminal(false, true, 29.23),
                null
        );

        float[] result = controller.normalize14Dimensions(request);

        assertEquals(14, result.length);
        assertEquals(0.0041f, result[0], 0.001);
        assertEquals(0.1667f, result[1], 0.001);
        assertEquals(0.05f, result[2], 0.001);
        assertEquals(0.7826f, result[3], 0.01);
        assertEquals(0.3333f, result[4], 0.001);
        assertEquals(-1.0f, result[5], 0.001);
        assertEquals(-1.0f, result[6], 0.001);
        assertEquals(0.0292f, result[7], 0.001);
        assertEquals(0.15f, result[8], 0.001);
        assertEquals(0.0f, result[9], 0.001);
        assertEquals(1.0f, result[10], 0.001);
        assertEquals(0.0f, result[11], 0.001);
        assertEquals(0.15f, result[12], 0.001);
        assertEquals(0.006f, result[13], 0.001);
    }

    @Test
    void normalize_fraudulentTransaction_highValues() {
        FraudRequest request = new FraudRequest(
                "tx-3330991687",
                new FraudRequest.Transaction(9505.97, 10, LocalDateTime.of(2026, 3, 14, 5, 15, 12)),
                new FraudRequest.Customer(81.28, 20, List.of("MERC-008", "MERC-007", "MERC-005")),
                new FraudRequest.Merchant("MERC-068", "7802", 54.86),
                new FraudRequest.Terminal(false, true, 952.27),
                null
        );

        float[] result = controller.normalize14Dimensions(request);

        assertEquals(14, result.length);
        assertEquals(0.9506f, result[0], 0.001);
        assertEquals(0.8333f, result[1], 0.001);
        assertEquals(1.0f, result[2], 0.001);
        assertEquals(0.2174f, result[3], 0.01);
        assertEquals(0.8333f, result[4], 0.001);
        assertEquals(-1.0f, result[5], 0.001);
        assertEquals(-1.0f, result[6], 0.001);
        assertEquals(0.9523f, result[7], 0.001);
        assertEquals(1.0f, result[8], 0.001);
        assertEquals(0.0f, result[9], 0.001);
        assertEquals(1.0f, result[10], 0.001);
        assertEquals(1.0f, result[11], 0.001);
        assertEquals(0.75f, result[12], 0.001);
        assertEquals(0.0055f, result[13], 0.001);
    }

    @Test
    void normalize_clampsValuesAboveOne() {
        FraudRequest request = new FraudRequest(
                "tx-test",
                new FraudRequest.Transaction(15000.0, 20, LocalDateTime.of(2026, 3, 14, 5, 15, 12)),
                new FraudRequest.Customer(100.0, 30, List.of("MERC-001")),
                new FraudRequest.Merchant("MERC-099", "5411", 15000.0),
                new FraudRequest.Terminal(true, false, 2000.0),
                null
        );

        float[] result = controller.normalize14Dimensions(request);

        assertEquals(1.0f, result[0]);
        assertEquals(1.0f, result[1]);
        assertEquals(1.0f, result[7]);
        assertEquals(1.0f, result[8]);
        assertEquals(1.0f, result[13]);
    }

    @Test
    void normalize_knownMerchant_unknownMerchantFlag() {
        FraudRequest withKnown = new FraudRequest(
                "tx-test",
                new FraudRequest.Transaction(100.0, 1, LocalDateTime.of(2026, 3, 14, 12, 0, 0)),
                new FraudRequest.Customer(100.0, 1, List.of("MERC-001")),
                new FraudRequest.Merchant("MERC-001", "5411", 100.0),
                new FraudRequest.Terminal(true, true, 10.0),
                null
        );

        FraudRequest withUnknown = new FraudRequest(
                "tx-test",
                new FraudRequest.Transaction(100.0, 1, LocalDateTime.of(2026, 3, 14, 12, 0, 0)),
                new FraudRequest.Customer(100.0, 1, List.of("MERC-001")),
                new FraudRequest.Merchant("MERC-099", "5411", 100.0),
                new FraudRequest.Terminal(true, true, 10.0),
                null
        );

        float[] known = controller.normalize14Dimensions(withKnown);
        float[] unknown = controller.normalize14Dimensions(withUnknown);

        assertEquals(0.0f, known[11]);
        assertEquals(1.0f, unknown[11]);
    }
}

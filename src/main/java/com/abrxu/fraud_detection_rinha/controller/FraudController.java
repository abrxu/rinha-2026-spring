package com.abrxu.fraud_detection_rinha.controller;

import com.abrxu.fraud_detection_rinha.service.VectorStoreService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.time.temporal.ChronoUnit;

@RestController
public class FraudController {

    private static final double THRESHOLD = 0.6;

    private final VectorStoreService vectorStore;

    public FraudController(VectorStoreService vectorStore) {
        this.vectorStore = vectorStore;
    }

    @GetMapping("/ready")
    public ResponseEntity<Void> ready() {
        return ResponseEntity.ok().build();
    }

    @PostMapping("/fraud-score")
    public ResponseEntity<FraudResponse> detect(@RequestBody FraudRequest request) {
        float[] vector = normalize14Dimensions(request);
        double fraudScore = vectorStore.computeFraudScore(vector);
        boolean approved = fraudScore < THRESHOLD;
        return ResponseEntity.ok(new FraudResponse(approved, fraudScore));
    }

    float[] normalize14Dimensions(FraudRequest request) {
        float[] dims = new float[14];

        float maxAmount = vectorStore.getNormalizationConstant("max_amount");
        float maxInstallments = vectorStore.getNormalizationConstant("max_installments");
        float amountVsAvgRatio = vectorStore.getNormalizationConstant("amount_vs_avg_ratio");
        float maxMinutes = vectorStore.getNormalizationConstant("max_minutes");
        float maxKm = vectorStore.getNormalizationConstant("max_km");
        float maxTxCount24h = vectorStore.getNormalizationConstant("max_tx_count_24h");
        float maxMerchantAvgAmount = vectorStore.getNormalizationConstant("max_merchant_avg_amount");

        dims[0] = clamp(request.transaction().amount() / maxAmount);
        dims[1] = clamp((double) request.transaction().installments() / maxInstallments);
        dims[2] = request.customer().avg_amount() > 0.0
                ? clamp((request.transaction().amount() / request.customer().avg_amount()) / amountVsAvgRatio)
                : 1.0f;

        OffsetDateTime requestedAtUtc = request.transaction().requested_at().withOffsetSameInstant(ZoneOffset.UTC);
        dims[3] = (float) requestedAtUtc.getHour() / 23.0f;
        dims[4] = (float) (requestedAtUtc.getDayOfWeek().getValue() - 1) / 6.0f;

        if (request.last_transaction() != null) {
            long minutes = ChronoUnit.MINUTES.between(
                    request.last_transaction().timestamp().toInstant(),
                    request.transaction().requested_at().toInstant()
            );
            dims[5] = clamp((double) minutes / maxMinutes);
            dims[6] = clamp(request.last_transaction().km_from_current() / maxKm);
        } else {
            dims[5] = -1.0f;
            dims[6] = -1.0f;
        }

        dims[7] = clamp(request.terminal().km_from_home() / maxKm);
        dims[8] = clamp((double) request.customer().tx_count_24h() / maxTxCount24h);
        dims[9] = request.terminal().is_online() ? 1.0f : 0.0f;
        dims[10] = request.terminal().card_present() ? 1.0f : 0.0f;
        dims[11] = request.customer().known_merchants().contains(request.merchant().id()) ? 0.0f : 1.0f;
        dims[12] = vectorStore.getMccRisk(request.merchant().mcc());
        dims[13] = clamp(request.merchant().avg_amount() / maxMerchantAvgAmount);

        return dims;
    }

    private static float clamp(double value) {
        if (value <= 0.0) return 0.0f;
        if (value >= 1.0) return 1.0f;
        return (float) value;
    }
}

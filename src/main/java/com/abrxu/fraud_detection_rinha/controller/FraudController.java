package com.abrxu.fraud_detection_rinha.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;

@RestController
public class FraudController {

    private static final int MAX_AMOUNT = 1000;
    private static final int MAX_INSTALLMENTS = 1000;
    private static final int AMOUNT_VS_AVG_RATIO = 10;

    private static final int MAX_MINUTES = 1440;
    private static final double MAX_KM = 1000;

    private static final int MAX_TX_COUNT_24H = 20;
    private static final int MAX_MERCHANT_AVG_AMOUNT = 10000;

    @GetMapping("/ready")
    public ResponseEntity<Void> ready() {
        return ResponseEntity.ok().build();
    }

    @PostMapping("/fraud-score")
    public ResponseEntity<FraudResponse> detect(@RequestBody FraudRequest request) {
        List<Number> normalizedDimensions = normalize14Dimensions(request);

        return ResponseEntity.ok(new FraudResponse(true, 2.0));
    }

    public List<Number> normalize14Dimensions(FraudRequest request) {
        List<Number> response = new ArrayList<>();

        List<Double> valuesFromTransactionNormalization = transactionNormalization(request);
        response.addAll(valuesFromTransactionNormalization);

        List<Double> valuesFromLastTransaction = (request.last_transaction() != null)
                ? (lastTransactionsNormalization(request))
                : List.of(-1.0, -1.0);
        response.addAll(valuesFromLastTransaction);

        double normalizedKmFromHome = limitDouble(request.terminal().km_from_home(), MAX_KM);
        response.add(normalizedKmFromHome);

        double normalizedLast24HoursTransactions = limitDouble(request.customer().tx_count_24h(), MAX_TX_COUNT_24H);
        response.add(normalizedLast24HoursTransactions);

        int terminalIsOnline = request.terminal().is_online() ? 1 : 0;
        response.add(terminalIsOnline);

        int cardIsPresent = request.terminal().card_present() ? 1 : 0;
        response.add(cardIsPresent);

        int unknwonMerchant = request.customer().known_merchants()
                .contains(request.merchant().id()) ? 0 : 1;
        response.add(unknwonMerchant);

        double mccRisk = 0.5; // TODO: PEGAR DO mcc_risk.json, valor default é 0.5 caso não estiver lá
        response.add(mccRisk);

        double merchantAverageAmount = limitDouble(request.merchant().avg_amount(), MAX_MERCHANT_AVG_AMOUNT);
        response.add(merchantAverageAmount);

        return response;
    }

    public List<Double> transactionNormalization(FraudRequest request) {
        double normalizedAmount = limitDouble(request.transaction().amount(), MAX_AMOUNT);
        double normalizedInstallments = limitInt(request.transaction().installments(), MAX_INSTALLMENTS);
        double normalizedAmountVersusAverage = limitDouble(
                (request.transaction().amount() / request.customer().avg_amount()),
                AMOUNT_VS_AVG_RATIO
        );
        double normalizedDayHour = request.transaction().requested_at().getHour() / 23.0;
        double normalizedDayOfWeek = (request.transaction().requested_at().getDayOfWeek().getValue() - 1) / 6.0;

        return List.of(
                normalizedAmount,
                normalizedInstallments,
                normalizedAmountVersusAverage,
                normalizedDayHour,
                normalizedDayOfWeek
        );
    }

    public List<Double> lastTransactionsNormalization(FraudRequest request) {
        int minutesFromLastTransaction = (int) ChronoUnit.MINUTES
                .between(request.transaction().requested_at(), request.last_transaction().timestamp());

        return List.of(
                limitInt(minutesFromLastTransaction, MAX_MINUTES),
                limitDouble(request.last_transaction().km_from_current(), MAX_KM)
        );
    }

    public double limitDouble(double requested, double limit) {
        double limitedValue = (requested / limit);

        if (limitedValue >= 1.0) return 1.0;
        if (limitedValue <= 0) return 0.0;

        return limitedValue;
    }

    public double limitInt(int requested, int limit) {
        int limitedValue = (requested / limit);

        if (limitedValue >= 1) return 1;
        if (limitedValue <= 0) return 0;

        return limitedValue;
    }

}



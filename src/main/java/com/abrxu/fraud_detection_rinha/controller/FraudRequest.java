package com.abrxu.fraud_detection_rinha.controller;

import com.fasterxml.jackson.annotation.JsonFormat;

import java.time.OffsetDateTime;
import java.util.List;

public record FraudRequest(
        String id,
        Transaction transaction,
        Customer customer,
        Merchant merchant,
        Terminal terminal,
        LastTransaction last_transaction
) {
    public record Transaction(
            double amount,
            int installments,
            @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss[.SSS]X") OffsetDateTime requested_at
    ) {}

    public record Customer(
            double avg_amount,
            int tx_count_24h,
            List<String> known_merchants
    ) {}

    public record Merchant(
            String id,
            String mcc,
            double avg_amount
    ) {}

    public record Terminal(
            boolean is_online,
            boolean card_present,
            double km_from_home
    ) {}

    public record LastTransaction(
            @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss[.SSS]X") OffsetDateTime timestamp,
            double km_from_current
    ) {}
}

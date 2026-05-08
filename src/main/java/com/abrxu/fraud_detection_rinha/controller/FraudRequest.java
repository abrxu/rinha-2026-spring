package com.abrxu.fraud_detection_rinha.controller;

import java.time.LocalDateTime;
import java.util.List;

// TODO: ver se wrapper classes ou primitivos vão ser mais performáticos
public record FraudRequest(
        String id,
        Transaction transaction,
        Customer customer,
        Merchant merchant,
        Terminal terminal,
        LastTransaction last_transaction
) {

}

record Transaction(
        double amount,
        int installments,
        LocalDateTime requested_at
) {
}

record Customer(
        double avg_amount,
        int tx_count_24h,
        List<String> known_merchants // TODO: ver se List vai ser o mais performático
) {
}

record Merchant(
        String id,
        String mcc,
        double avg_amount
        ) {
}

record Terminal(
        boolean is_online,
        boolean card_present,
        double km_from_home
) {
}

record LastTransaction(
        LocalDateTime timestamp,
        double km_from_current
) {
}
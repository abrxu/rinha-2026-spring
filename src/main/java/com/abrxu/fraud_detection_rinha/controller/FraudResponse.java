package com.abrxu.fraud_detection_rinha.controller;

public record FraudResponse(
        boolean approved,
        double fraud_score
) {}

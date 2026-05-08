package com.abrxu.fraud_detection_rinha;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import com.abrxu.fraud_detection_rinha.service.VectorStoreService;

@SpringBootTest
class FraudDetectionRinhaApplicationTests {

	@MockitoBean
	private VectorStoreService vectorStoreService;

	@Test
	void contextLoads() {
	}

}

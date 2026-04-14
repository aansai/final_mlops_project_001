# ═══════════════════════════════════════════════════════════════
# test.py — MLOPS Clothes Project
# ═══════════════════════════════════════════════════════════════
# Run with: pytest test.py -v
#
# ✅ NEVER CHANGE:
#   - client setup with TestClient
#   - mock model/processor pattern
#   - test function naming (test_ prefix)
#   - status code assertions
#
# 🔄 ALWAYS CHANGE:
#   - VALID_PAYLOAD → your actual feature fields
#   - field names in invalid payload tests
#   - predicted field name (predicted_profit → your target)
#   - price_tier valid/invalid values → your categories
# ═══════════════════════════════════════════════════════════════

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient


# ───────────────────────────────────────────
# Mock model and processor
# ✅ NEVER CHANGE: mock pattern stays same
# We mock joblib.load so tests don't need
# actual model.pkl / processor.pkl files
# ───────────────────────────────────────────
mock_model = MagicMock()
mock_model.predict.return_value = np.array([1234.56])

mock_processor = MagicMock()
mock_processor.transform.return_value = np.zeros((1, 10))

# ✅ NEVER CHANGE: patch joblib before importing app
with patch("joblib.load", side_effect=[mock_model, mock_processor]):
    from app import app, MODEL_STORE
    MODEL_STORE["model"]      = mock_model
    MODEL_STORE["processor"]  = mock_processor
    MODEL_STORE["model_name"] = "XGBRegressor"

# ✅ NEVER CHANGE: TestClient setup
client = TestClient(app)


# ───────────────────────────────────────────
# 🔄 CHANGE: update all fields to match YOUR
# PredictRequest schema in app.py
# ───────────────────────────────────────────
VALID_PAYLOAD = {
    "Product_Category" : "Shirts",
    "Product_Name"     : "Polo Shirt",
    "City"             : "Mumbai",
    "Segment"          : "Consumer",
    "price_tier"       : "mid",
    "Units_Sold"       : 10.0,
    "Unit_Price"       : 499.0,
    "Discount_%"       : 0.1,
    "Sales_Amount"     : 4491.0,
    "is_return"        : 0,
    "is_bulk_order"    : 1,
    "order_month"      : 6,
    "order_quarter"    : 2,
    "order_dayofweek"  : 3,
    "is_zero_sale"     : 0,
    "discount_applied" : 1,
}


# ═══════════════════════════════════════════════
# HEALTH TESTS
# ✅ NEVER CHANGE these tests
# ═══════════════════════════════════════════════

def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_status():
    response = client.get("/health")
    assert response.json()["status"] == "ok"


def test_health_shows_model_loaded():
    response = client.get("/health")
    assert response.json()["model_loaded"] is True


def test_root_returns_200():
    response = client.get("/")
    assert response.status_code == 200


def test_model_info_returns_200():
    response = client.get("/model-info")
    assert response.status_code == 200


def test_model_info_has_model_name():
    response = client.get("/model-info")
    assert "model_name" in response.json()


# ═══════════════════════════════════════════════
# PREDICT — HAPPY PATH TESTS
# ✅ NEVER CHANGE: test structure
# 🔄 CHANGE: "predicted_profit" → your target field name
# ═══════════════════════════════════════════════

def test_predict_returns_200():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_predict_returns_predicted_profit():
    # 🔄 CHANGE: predicted_profit → your target column name
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert "predicted_profit" in response.json()


def test_predict_profit_is_float():
    # 🔄 CHANGE: predicted_profit → your target column name
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert isinstance(response.json()["predicted_profit"], float)


def test_predict_returns_request_id():
    # ✅ NEVER CHANGE: request_id always in response
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert "request_id" in response.json()


def test_predict_returns_model_name():
    # ✅ NEVER CHANGE: model_name always in response
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert "model_name" in response.json()


def test_predict_returns_latency():
    # ✅ NEVER CHANGE: latency_ms always in response
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert "latency_ms" in response.json()


def test_predict_status_is_success():
    # ✅ NEVER CHANGE
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.json()["status"] == "success"


# ═══════════════════════════════════════════════
# PREDICT — VALIDATION TESTS
# ✅ NEVER CHANGE: test structure (422 = validation error)
# 🔄 CHANGE: field names and invalid values to match YOUR schema
# ═══════════════════════════════════════════════

def test_predict_rejects_invalid_price_tier():
    # 🔄 CHANGE: price_tier → your categorical field
    # 🔄 CHANGE: "WRONG" → any value not in your validator
    payload = {**VALID_PAYLOAD, "price_tier": "WRONG_VALUE"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_negative_unit_price():
    # 🔄 CHANGE: Unit_Price → your field with gt=0 constraint
    payload = {**VALID_PAYLOAD, "Unit_Price": -100.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_discount_above_1():
    # 🔄 CHANGE: Discount_% → your field with le=1.0 constraint
    payload = {**VALID_PAYLOAD, "Discount_%": 1.5}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_invalid_month():
    # 🔄 CHANGE: order_month → your field with ge=1, le=12
    payload = {**VALID_PAYLOAD, "order_month": 13}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_blank_city():
    # 🔄 CHANGE: City → your string field with non_empty validator
    payload = {**VALID_PAYLOAD, "City": "   "}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    # ✅ NEVER CHANGE: missing required fields always = 422
    payload = {**VALID_PAYLOAD}
    del payload["Units_Sold"]   # 🔄 CHANGE: field name to any required field
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_empty_body():
    # ✅ NEVER CHANGE
    response = client.post("/predict", json={})
    assert response.status_code == 422


# ═══════════════════════════════════════════════
# PREDICT — BOUNDARY TESTS
# 🔄 CHANGE: all field names and boundary values
# ═══════════════════════════════════════════════

def test_predict_accepts_zero_discount():
    # Discount_% = 0.0 should be valid (ge=0.0)
    payload = {**VALID_PAYLOAD, "Discount_%": 0.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_accepts_full_discount():
    # Discount_% = 1.0 should be valid (le=1.0)
    payload = {**VALID_PAYLOAD, "Discount_%": 1.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_accepts_all_price_tiers():
    # 🔄 CHANGE: list all valid values from your VALID_PRICE_TIERS
    valid_tiers = ["budget", "mid", "upper-mid", "premium"]
    for tier in valid_tiers:
        payload = {**VALID_PAYLOAD, "price_tier": tier}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, f"price_tier='{tier}' should be valid"
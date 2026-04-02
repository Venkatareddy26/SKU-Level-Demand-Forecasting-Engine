# API Specification - SKU Demand Forecasting Engine

## Overview
RESTful API for SKU-level demand forecasting with authentication, rate limiting, and comprehensive error handling.

**Base URL**: `https://api.demandforecast.ai/v1`

---

## Authentication

All API requests require authentication using API keys.

### Headers
```
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

### Get API Key
```http
POST /auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "company": "ABC Distributors",
  "plan": "starter"
}

Response:
{
  "api_key": "sk_live_abc123...",
  "plan": "starter",
  "rate_limit": 1000,
  "expires_at": "2025-04-02T00:00:00Z"
}
```

---

## Endpoints

### 1. Health Check

```http
GET /health

Response:
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-04-02T19:15:00Z"
}
```

---

### 2. Generate Forecast

Generate demand forecast for a specific SKU.

```http
POST /forecast
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json

{
  "sku_id": "SKU_001",
  "forecast_weeks": 8,
  "historical_data": [
    {
      "date": "2024-01-01",
      "sales": 120,
      "price": 99.99
    },
    {
      "date": "2024-01-02",
      "sales": 135,
      "price": 99.99
    }
  ],
  "include_drivers": true,
  "confidence_level": 0.95
}

Response:
{
  "sku_id": "SKU_001",
  "forecast_weeks": 8,
  "generated_at": "2024-04-02T19:15:00Z",
  "forecast": [
    {
      "date": "2024-04-03",
      "predicted_sales": 142.5,
      "lower_bound": 114.0,
      "upper_bound": 171.0
    },
    ...
  ],
  "summary": {
    "avg_weekly_demand": 998.5,
    "total_demand": 7988,
    "reorder_point": 1997,
    "confidence_level": 0.95
  },
  "drivers": [
    {
      "feature": "Festival Week",
      "impact": 340,
      "direction": "increase",
      "explanation": "Festival week increasing demand by 340 units"
    },
    ...
  ],
  "metrics": {
    "wrmsse": 0.55,
    "mape": 21.3,
    "mae": 1.45
  }
}
```

---

### 3. Batch Forecast

Generate forecasts for multiple SKUs in one request.

```http
POST /forecast/batch
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json

{
  "skus": ["SKU_001", "SKU_002", "SKU_003"],
  "forecast_weeks": 8,
  "data_source": "uploaded_csv",
  "csv_url": "https://yourbucket.s3.amazonaws.com/sales_data.csv"
}

Response:
{
  "batch_id": "batch_abc123",
  "status": "processing",
  "total_skus": 3,
  "estimated_completion": "2024-04-02T19:20:00Z",
  "results_url": "/forecast/batch/batch_abc123"
}
```

---

### 4. Get Batch Results

```http
GET /forecast/batch/{batch_id}
Authorization: Bearer YOUR_API_KEY

Response:
{
  "batch_id": "batch_abc123",
  "status": "completed",
  "total_skus": 3,
  "completed_skus": 3,
  "failed_skus": 0,
  "results": [
    {
      "sku_id": "SKU_001",
      "forecast": [...],
      "summary": {...}
    },
    ...
  ],
  "download_url": "https://api.demandforecast.ai/v1/download/batch_abc123.csv"
}
```

---

### 5. Upload Historical Data

Upload historical sales data for future forecasting.

```http
POST /data/upload
Authorization: Bearer YOUR_API_KEY
Content-Type: multipart/form-data

file: sales_data.csv

Response:
{
  "data_id": "data_xyz789",
  "filename": "sales_data.csv",
  "rows": 10000,
  "skus": 50,
  "date_range": {
    "start": "2023-01-01",
    "end": "2024-12-31"
  },
  "status": "validated",
  "errors": []
}
```

---

### 6. Get Demand Drivers

Get explainable demand drivers for a specific SKU.

```http
GET /drivers/{sku_id}
Authorization: Bearer YOUR_API_KEY

Query Parameters:
- date: YYYY-MM-DD (optional, defaults to latest)
- top_n: integer (default: 5)

Response:
{
  "sku_id": "SKU_001",
  "date": "2024-04-02",
  "drivers": [
    {
      "rank": 1,
      "feature": "Festival Week",
      "value": 1,
      "impact": 340,
      "direction": "increase",
      "explanation": "Festival week increasing demand by 340 units"
    },
    ...
  ]
}
```

---

### 7. Model Performance

Get model performance metrics.

```http
GET /model/performance
Authorization: Bearer YOUR_API_KEY

Response:
{
  "model_version": "1.0.0",
  "trained_on": "2024-03-15T00:00:00Z",
  "metrics": {
    "wrmsse": 0.55,
    "mape": 21.3,
    "mae": 1.45,
    "rmse": 2.78
  },
  "benchmark": {
    "dataset": "M5 Forecasting",
    "skus": 42840,
    "baseline_wrmsse": 0.90
  }
}
```

---

### 8. Reorder Point Calculation

Calculate optimal reorder point for inventory management.

```http
POST /inventory/reorder-point
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json

{
  "sku_id": "SKU_001",
  "lead_time_days": 7,
  "service_level": 0.95,
  "safety_stock_multiplier": 1.5
}

Response:
{
  "sku_id": "SKU_001",
  "reorder_point": 1997,
  "safety_stock": 499,
  "avg_daily_demand": 142.5,
  "lead_time_demand": 997.5,
  "service_level": 0.95,
  "recommendation": "Reorder when inventory falls below 1997 units"
}
```

---

## Error Responses

### Standard Error Format
```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Missing required field: sku_id",
    "details": {
      "field": "sku_id",
      "expected": "string"
    },
    "request_id": "req_abc123"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| INVALID_REQUEST | 400 | Malformed request or missing fields |
| UNAUTHORIZED | 401 | Invalid or missing API key |
| FORBIDDEN | 403 | API key lacks required permissions |
| NOT_FOUND | 404 | Resource not found |
| RATE_LIMIT_EXCEEDED | 429 | Too many requests |
| INTERNAL_ERROR | 500 | Server error |
| MODEL_ERROR | 503 | Model unavailable or failed |

---

## Rate Limits

### By Plan

| Plan | Requests/Hour | Requests/Day | SKUs/Request |
|------|---------------|--------------|--------------|
| Starter | 100 | 1,000 | 10 |
| Growth | 500 | 5,000 | 50 |
| Enterprise | 2,000 | 20,000 | Unlimited |

### Rate Limit Headers
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1712087400
```

---

## Webhooks

Subscribe to events for async notifications.

### Configure Webhook
```http
POST /webhooks
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json

{
  "url": "https://yourdomain.com/webhook",
  "events": ["forecast.completed", "batch.completed"],
  "secret": "whsec_abc123"
}

Response:
{
  "webhook_id": "wh_xyz789",
  "url": "https://yourdomain.com/webhook",
  "events": ["forecast.completed", "batch.completed"],
  "status": "active"
}
```

### Webhook Payload
```json
{
  "event": "forecast.completed",
  "timestamp": "2024-04-02T19:15:00Z",
  "data": {
    "sku_id": "SKU_001",
    "forecast_weeks": 8,
    "status": "success",
    "results_url": "/forecast/SKU_001/latest"
  }
}
```

---

## SDK Examples

### Python
```python
import requests

API_KEY = "sk_live_abc123..."
BASE_URL = "https://api.demandforecast.ai/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Generate forecast
response = requests.post(
    f"{BASE_URL}/forecast",
    headers=headers,
    json={
        "sku_id": "SKU_001",
        "forecast_weeks": 8,
        "historical_data": [...]
    }
)

forecast = response.json()
print(f"Predicted demand: {forecast['summary']['total_demand']}")
```

### JavaScript
```javascript
const API_KEY = 'sk_live_abc123...';
const BASE_URL = 'https://api.demandforecast.ai/v1';

const response = await fetch(`${BASE_URL}/forecast`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${API_KEY}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    sku_id: 'SKU_001',
    forecast_weeks: 8,
    historical_data: [...]
  })
});

const forecast = await response.json();
console.log(`Predicted demand: ${forecast.summary.total_demand}`);
```

### cURL
```bash
curl -X POST https://api.demandforecast.ai/v1/forecast \
  -H "Authorization: Bearer sk_live_abc123..." \
  -H "Content-Type: application/json" \
  -d '{
    "sku_id": "SKU_001",
    "forecast_weeks": 8,
    "historical_data": [...]
  }'
```

---

## Versioning

API versions are specified in the URL path: `/v1/`, `/v2/`, etc.

- Current version: `v1`
- Deprecation notice: 6 months before sunset
- Breaking changes: New major version

---

## Support

- Documentation: https://docs.demandforecast.ai
- Status page: https://status.demandforecast.ai
- Support email: support@demandforecast.ai
- Enterprise support: Available for Enterprise plan

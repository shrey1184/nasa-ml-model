# NASA Climate Model - Deployment Guide

Complete guide for deploying the NASA Climate LSTM model as a production API.

## 📦 What's Included

```
deployment/
├── export_for_deployment.py     # Export script to create bundle
├── model_bundle/                # Auto-generated bundle (after export)
│   ├── climate_lstm_model.keras
│   ├── lstm_scaler.pkl
│   ├── cleaning_scalers.pkl
│   ├── cleaning_encoders.pkl
│   ├── model_configuration.json
│   ├── deployment_metadata.json
│   └── requirements_deployment.txt
├── api/
│   ├── api_server.py            # FastAPI production server
│   └── test_api_client.py       # API test client
├── Dockerfile                    # Docker image definition
├── docker-compose.yml           # Docker Compose configuration
└── README.md                    # This file
```

## 🚀 Quick Start

### Step 1: Export Model Bundle

```bash
# Activate virtual environment
cd "/home/shrey/Documents/ml nasa"
source venv/bin/activate

# Export model bundle for deployment
python deployment/export_for_deployment.py
```

This creates `deployment/model_bundle/` with all necessary files.

### Step 2: Test Locally

```bash
# Run API server
python deployment/api/api_server.py

# In another terminal, test the API
python deployment/api/test_api_client.py
```

### Step 3: Deploy with Docker (Recommended)

```bash
cd deployment/

# Build and run with Docker Compose
docker-compose up --build

# Or build Docker image manually
docker build -t nasa-climate-api .
docker run -p 8000:8000 nasa-climate-api
```

## 🌐 API Endpoints

### Health Check
```bash
GET http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Model Information
```bash
GET http://localhost:8000/model-info
```

Response includes model architecture, features, and performance metrics.

### Make Prediction
```bash
POST http://localhost:8000/predict
Content-Type: application/json

{
  "latitude": 28.6139,
  "longitude": 77.2090,
  "months_back": 24
}
```

Response:
```json
{
  "location": {
    "latitude": 28.6139,
    "longitude": 77.2090
  },
  "data_fetched": true,
  "samples": 24,
  "predictions": [
    {
      "date": "2023-11",
      "temperature_anomaly": 0.0674,
      "precipitation_anomaly": -0.0935,
      "actual_temperature": 24.5,
      "actual_precipitation": 12.3
    }
    // ... more predictions
  ],
  "summary": {
    "temperature_anomaly": {
      "mean": 0.0234,
      "std": 0.1456,
      "min": -0.2516,
      "max": 0.2361
    },
    "precipitation_anomaly": {
      "mean": -0.0567,
      "std": 0.4213,
      "min": -1.4285,
      "max": 2.1598
    }
  },
  "timestamp": "2025-11-07T18:45:00"
}
```

## 📋 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔄 How It Works

### Prediction Workflow

1. **Receive Request** → API receives location (lat, lon) and time range
2. **Fetch NASA Data** → Get real-time climate data from NASA POWER API
3. **Preprocess** → Clean and transform data using saved scalers/encoders
4. **Feature Engineering** → Calculate derived features (T2M_range, heat_index, etc.)
5. **Scale Features** → Apply StandardScaler to match training distribution
6. **Predict** → Run LSTM model to get anomaly predictions
7. **Return Results** → Send predictions and summary statistics

### Data Pipeline

```
NASA POWER API → Raw Data → Cleaning → Feature Engineering → Scaling → LSTM Model → Predictions
                    ↓           ↓              ↓               ↓            ↓
              cleaning_scalers  derived     lstm_scaler    predictions   JSON Response
              cleaning_encoders features
```

## 🛠️ Configuration

### Environment Variables

- `MODEL_BUNDLE_DIR`: Path to model bundle (default: `./model_bundle`)
- `API_HOST`: Server host (default: `0.0.0.0`)
- `API_PORT`: Server port (default: `8000`)

### Model Bundle Contents

The exported bundle includes:

1. **climate_lstm_model.keras** (3.79 MB)
   - Trained LSTM model
   - Input: 18 features
   - Output: 2 targets (temperature & precipitation anomalies)

2. **lstm_scaler.pkl**
   - StandardScaler for model input features
   - Fitted on training data

3. **cleaning_scalers.pkl**
   - Scalers for raw data preprocessing
   - Applied before feature engineering

4. **cleaning_encoders.pkl**
   - Encoders for categorical variables
   - Used for season encoding

5. **model_configuration.json**
   - Feature definitions
   - Required input features list
   - Model settings

6. **deployment_metadata.json**
   - Model performance metrics
   - API requirements
   - Deployment information

## 🧪 Testing

### Manual Testing

```bash
# Test with curl
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 28.6139,
    "longitude": 77.2090,
    "months_back": 24
  }'
```

### Automated Testing

```bash
python deployment/api/test_api_client.py
```

Tests multiple locations and validates responses.

### Load Testing

```bash
# Install locust
pip install locust

# Run load test
locust -f deployment/api/load_test.py --host http://localhost:8000
```

## 📊 Monitoring

### Logs

The API logs all requests and errors to stdout. Configure log level in `api_server.py`:

```python
logging.basicConfig(level=logging.INFO)  # or DEBUG, WARNING, ERROR
```

### Metrics

Consider adding:
- Request count
- Response time
- Error rate
- Model inference time

Use tools like Prometheus + Grafana for production monitoring.

## 🚀 Production Deployment

### Requirements

- Python 3.11+
- 2GB RAM minimum (4GB recommended)
- Internet access (for NASA API calls)
- 100MB disk space for model bundle

### Cloud Deployment Options

#### 1. AWS (EC2 + ECS)
```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag nasa-climate-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/nasa-climate-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/nasa-climate-api:latest
```

#### 2. Google Cloud (Cloud Run)
```bash
# Deploy to Cloud Run
gcloud run deploy nasa-climate-api \
  --image gcr.io/<project-id>/nasa-climate-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### 3. Azure (Container Instances)
```bash
az container create \
  --resource-group myResourceGroup \
  --name nasa-climate-api \
  --image <registry>/nasa-climate-api:latest \
  --dns-name-label nasa-climate-api \
  --ports 8000
```

#### 4. Heroku
```bash
heroku container:push web --app nasa-climate-api
heroku container:release web --app nasa-climate-api
```

### Scaling

For high traffic, consider:
- Load balancer (NGINX, HAProxy)
- Multiple API instances
- Redis cache for frequent locations
- Database for storing predictions

## 🔒 Security

### Best Practices

1. **API Authentication** - Add API key or OAuth
2. **Rate Limiting** - Prevent abuse
3. **Input Validation** - Validate lat/lon ranges
4. **HTTPS** - Use SSL certificates
5. **CORS** - Configure allowed origins

### Example: Add API Key

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-api-key"
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
```

## 🐛 Troubleshooting

### Model not loading
- Check model bundle path is correct
- Verify all files exist in `model_bundle/`
- Check TensorFlow version compatibility

### NASA API timeout
- Increase request timeout
- Add retry logic
- Cache responses for common locations

### Memory issues
- Reduce batch size in predictions
- Use gunicorn workers with memory limits
- Monitor memory usage

## 📝 Performance Optimization

1. **Caching** - Cache NASA API responses
2. **Batch Processing** - Process multiple locations together
3. **Async Operations** - Use async for NASA API calls
4. **Model Optimization** - Convert to TensorFlow Lite
5. **CDN** - Serve static assets via CDN

## 🔄 Updates

### Updating the Model

1. Retrain model with new data
2. Run `export_for_deployment.py` again
3. Replace model bundle in deployment
4. Restart API server
5. Verify with tests

### Version Control

Tag model versions in `deployment_metadata.json`:
```json
{
  "version": "1.1.0",
  "update_date": "2025-11-07",
  "changes": "Improved precipitation predictions"
}
```

## 📞 Support

For issues or questions:
- Check logs: `docker logs nasa-climate-api`
- Test health endpoint: `curl http://localhost:8000/health`
- Review API docs: `http://localhost:8000/docs`

## 📄 License

Same as main project.

---

**Last Updated:** November 7, 2025  
**API Version:** 1.0.0  
**Model Version:** 1.0.0

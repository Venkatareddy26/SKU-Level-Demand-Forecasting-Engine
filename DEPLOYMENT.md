# Deployment Guide

## Deployment Options

### 1. Streamlit Cloud (Easiest)

#### Steps:
1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect GitHub repository
4. Deploy with one click

#### Configuration:
- Main file: `app.py`
- Python version: 3.10
- Requirements: `requirements.txt`

#### Limitations:
- Free tier: 1GB RAM, limited compute
- Public URL (can be password-protected)

---

### 2. AWS Deployment

#### Option A: EC2 Instance

```bash
# Launch EC2 instance (t3.medium or larger)
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install dependencies
sudo apt update
sudo apt install python3-pip python3-venv nginx

# Clone repository
git clone your-repo-url
cd SKU-Level-Demand-Forecasting-Engine

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run with systemd
sudo nano /etc/systemd/system/forecast-app.service
```

**Service file:**
```ini
[Unit]
Description=SKU Forecast Dashboard
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/SKU-Level-Demand-Forecasting-Engine
Environment="PATH=/home/ubuntu/SKU-Level-Demand-Forecasting-Engine/venv/bin"
ExecStart=/home/ubuntu/SKU-Level-Demand-Forecasting-Engine/venv/bin/streamlit run app.py --server.port 8501

[Install]
WantedBy=multi-user.target
```

```bash
# Start service
sudo systemctl start forecast-app
sudo systemctl enable forecast-app

# Setup Nginx reverse proxy
sudo nano /etc/nginx/sites-available/forecast
```

**Nginx config:**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/forecast /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

#### Option B: AWS Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p python-3.10 forecast-app

# Create environment
eb create forecast-env

# Deploy
eb deploy
```

---

### 3. Docker Deployment

#### Dockerfile:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8501

# Run app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Build and run:
```bash
# Build image
docker build -t sku-forecast .

# Run container
docker run -p 8501:8501 sku-forecast

# Or use docker-compose
docker-compose up
```

#### docker-compose.yml:
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - PYTHONUNBUFFERED=1
```

---

### 4. Azure Deployment

#### Azure App Service:
```bash
# Install Azure CLI
az login

# Create resource group
az group create --name forecast-rg --location eastus

# Create App Service plan
az appservice plan create --name forecast-plan --resource-group forecast-rg --sku B1 --is-linux

# Create web app
az webapp create --resource-group forecast-rg --plan forecast-plan --name sku-forecast-app --runtime "PYTHON:3.10"

# Deploy code
az webapp up --name sku-forecast-app --resource-group forecast-rg
```

---

### 5. Google Cloud Platform

#### Cloud Run:
```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/sku-forecast

# Deploy to Cloud Run
gcloud run deploy sku-forecast \
  --image gcr.io/PROJECT_ID/sku-forecast \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## API Deployment (FastAPI)

### Create API wrapper:

**api.py:**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.models import LightGBMForecaster
from src.features import FeatureEngineer

app = FastAPI(title="SKU Forecast API")

# Load model at startup
model = LightGBMForecaster()
model.load("models/lightgbm_model.pkl")
fe = FeatureEngineer()

class ForecastRequest(BaseModel):
    sku_id: str
    forecast_weeks: int = 8
    historical_data: list

@app.post("/predict")
async def predict(request: ForecastRequest):
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.historical_data)
        
        # Feature engineering
        df_features = fe.build_features(df)
        
        # Predict
        predictions = model.predict(df_features)
        
        return {
            "sku_id": request.sku_id,
            "forecast": predictions.tolist(),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### Deploy API:
```bash
# Run locally
uvicorn api:app --host 0.0.0.0 --port 8000

# Deploy to AWS Lambda (using Mangum)
pip install mangum
```

---

## Production Checklist

### Security
- [ ] Enable HTTPS/SSL
- [ ] Add authentication (OAuth, API keys)
- [ ] Rate limiting
- [ ] Input validation
- [ ] CORS configuration
- [ ] Environment variables for secrets

### Performance
- [ ] Model caching
- [ ] Database for historical data
- [ ] CDN for static assets
- [ ] Load balancing
- [ ] Auto-scaling

### Monitoring
- [ ] Application logs
- [ ] Error tracking (Sentry)
- [ ] Performance monitoring (New Relic, DataDog)
- [ ] Uptime monitoring
- [ ] Cost tracking

### Backup
- [ ] Model versioning
- [ ] Data backups
- [ ] Disaster recovery plan

---

## Cost Estimates

### Streamlit Cloud
- Free tier: $0/month (limited)
- Pro: $250/month

### AWS
- EC2 t3.medium: ~$30/month
- Load balancer: ~$20/month
- Storage: ~$5/month
- **Total: ~$55/month**

### Azure
- App Service B1: ~$55/month

### GCP
- Cloud Run: ~$10-50/month (pay per use)

---

## Scaling Strategy

### Phase 1: MVP (0-100 users)
- Single EC2 instance
- Streamlit dashboard
- Manual model updates

### Phase 2: Growth (100-1000 users)
- Load balancer + 2-3 EC2 instances
- PostgreSQL database
- Automated model retraining
- API endpoints

### Phase 3: Scale (1000+ users)
- Kubernetes cluster
- Microservices architecture
- Real-time predictions
- Multi-region deployment
- CDN for global access

---

## Support & Maintenance

### Weekly Tasks
- Monitor error logs
- Check model performance
- Update festival calendar

### Monthly Tasks
- Retrain models with new data
- Performance optimization
- Security updates

### Quarterly Tasks
- Feature releases
- Infrastructure review
- Cost optimization

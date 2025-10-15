# MTQuant Deployment Guide

Complete guide for deploying MTQuant trading system to production.

## 📋 Prerequisites

### System Requirements
- **OS**: Ubuntu 22.04 LTS (recommended) or similar Linux distribution
- **CPU**: 8+ cores (for parallel training)
- **RAM**: 32GB+ (16GB minimum)
- **Storage**: 500GB+ SSD
- **Docker**: 24.0+
- **Docker Compose**: 2.20+

### Network Requirements
- **Firewall**: Allow ports 80 (HTTP), 443 (HTTPS), 8000 (API)
- **Broker Access**: Stable connection to MT4/MT5 servers
- **Internet**: Required for market data fetching

---

## 🚀 Quick Start (Docker)

### 1. Clone Repository
```bash
git clone https://github.com/your-org/mtquant.git
cd mtquant
```

### 2. Configure Environment
```bash
cp .env.example .env
nano .env
```

Required environment variables:
```env
# Database passwords
POSTGRES_PASSWORD=your_secure_password
QUESTDB_PASSWORD=your_secure_password

# MT5 Broker credentials
MT5_ACCOUNT=12345678
MT5_PASSWORD=your_broker_password
MT5_SERVER=YourBroker-Demo

# API Keys (if needed)
MARKET_DATA_API_KEY=your_api_key
```

### 3. Start Services
```bash
cd docker
docker-compose up -d
```

### 4. Verify Deployment
```bash
# Check all services are running
docker-compose ps

# Check logs
docker-compose logs -f backend

# Health check
curl http://localhost:8000/health
```

### 5. Access Application
- **Frontend**: http://localhost
- **API Docs**: http://localhost:8000/api/docs
- **QuestDB Console**: http://localhost:9000

---

## 🔧 Manual Deployment

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3-pip \
    postgresql-15 redis-server git curl build-essential
```

### 2. Setup PostgreSQL
```bash
sudo -u postgres psql

-- Create database and user
CREATE DATABASE mtquant;
CREATE USER mtquant_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE mtquant TO mtquant_user;
\q
```

### 3. Install QuestDB
```bash
# Download QuestDB
wget https://github.com/questdb/questdb/releases/download/7.3.1/questdb-7.3.1-rt-linux-amd64.tar.gz
tar xvf questdb-7.3.1-rt-linux-amd64.tar.gz
cd questdb

# Start QuestDB
./bin/questdb.sh start
```

### 4. Setup Python Environment
```bash
cd /opt/mtquant
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 5. Initialize Database
```bash
python scripts/init_database.py
```

### 6. Start Backend (Systemd)

Create `/etc/systemd/system/mtquant-backend.service`:
```ini
[Unit]
Description=MTQuant Backend API
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=mtquant
WorkingDirectory=/opt/mtquant
Environment="PATH=/opt/mtquant/venv/bin"
Environment="PYTHONPATH=/opt/mtquant"
ExecStart=/opt/mtquant/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable mtquant-backend
sudo systemctl start mtquant-backend
```

### 7. Setup Frontend (Nginx)

Build frontend:
```bash
cd frontend
npm install
npm run build
```

Nginx configuration `/etc/nginx/sites-available/mtquant`:
```nginx
server {
    listen 80;
    server_name your-domain.com;
    root /opt/mtquant/frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/mtquant /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## 🔒 Production Hardening

### 1. SSL/TLS (HTTPS)

Install Certbot:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 2. Firewall (UFW)
```bash
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable
```

### 3. Database Security

PostgreSQL (`/etc/postgresql/15/main/pg_hba.conf`):
```
# Only allow local connections
local   all             all                                     scram-sha-256
host    all             all             127.0.0.1/32            scram-sha-256
```

Redis (`/etc/redis/redis.conf`):
```
# Bind to localhost only
bind 127.0.0.1

# Require password
requirepass your_redis_password

# Disable dangerous commands
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command CONFIG ""
```

### 4. Application Secrets

Use environment variables or secrets management:
```bash
# Install Vault (optional)
wget https://releases.hashicorp.com/vault/1.15.0/vault_1.15.0_linux_amd64.zip
unzip vault_1.15.0_linux_amd64.zip
sudo mv vault /usr/local/bin/
```

---

## 📊 Monitoring & Logging

### 1. System Monitoring (Prometheus + Grafana)

Docker Compose addition:
```yaml
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
```

### 2. Log Aggregation (ELK Stack)
```bash
# TODO: Configure Elasticsearch, Logstash, Kibana
```

### 3. Application Logs
```bash
# View backend logs
sudo journalctl -u mtquant-backend -f

# Docker logs
docker-compose logs -f backend
```

---

## 🔄 Backup Strategy

### 1. Database Backups

PostgreSQL (daily cron):
```bash
#!/bin/bash
# /opt/mtquant/scripts/backup_postgres.sh
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
pg_dump -U mtquant_user mtquant | gzip > /backups/postgres_$TIMESTAMP.sql.gz

# Keep only last 30 days
find /backups -name "postgres_*.sql.gz" -mtime +30 -delete
```

QuestDB:
```bash
# Backup QuestDB data directory
tar -czf /backups/questdb_$(date +%Y%m%d).tar.gz /root/.questdb
```

### 2. Model Checkpoints
```bash
# Sync to S3 (AWS) or similar
aws s3 sync /opt/mtquant/models s3://your-bucket/models/
```

### 3. Configuration Backups
```bash
# Backup config files
tar -czf /backups/config_$(date +%Y%m%d).tar.gz /opt/mtquant/config
```

---

## 🚨 Troubleshooting

### Service Not Starting
```bash
# Check service status
sudo systemctl status mtquant-backend

# Check logs
sudo journalctl -u mtquant-backend -n 100

# Common issues:
# - Port already in use: lsof -i :8000
# - Database connection: psql -U mtquant_user -d mtquant
# - Permission issues: check file ownership
```

### High Memory Usage
```bash
# Check memory usage
free -h
docker stats

# Reduce workers if needed
# Edit docker-compose.yml: uvicorn --workers 2
```

### Database Connection Errors
```bash
# Test PostgreSQL connection
psql -h localhost -U mtquant_user -d mtquant

# Test QuestDB
curl http://localhost:9000/exec?query=SELECT+1

# Test Redis
redis-cli ping
```

---

## 📈 Scaling

### Horizontal Scaling (Multiple Workers)
```yaml
# docker-compose.yml
  backend:
    deploy:
      replicas: 3
    # Use load balancer (e.g., Nginx upstream)
```

### Vertical Scaling
- Increase Docker resource limits
- Add more CPU/RAM to server
- Use faster SSD storage

### Database Scaling
- PostgreSQL: Read replicas
- QuestDB: Partitioning by time
- Redis: Redis Cluster

---

## 🎯 Production Checklist

- [ ] SSL/TLS configured (HTTPS)
- [ ] Firewall rules in place
- [ ] Database passwords changed from defaults
- [ ] Automated backups configured
- [ ] Monitoring alerts set up
- [ ] Log rotation configured
- [ ] Health checks working
- [ ] Documentation updated
- [ ] Disaster recovery plan documented
- [ ] Team trained on deployment process

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [PostgreSQL Administration](https://www.postgresql.org/docs/15/admin.html)
- [QuestDB Documentation](https://questdb.io/docs/)
- [Nginx Documentation](https://nginx.org/en/docs/)

---

**For support:** Open an issue on GitHub or contact devops@mtquant.com



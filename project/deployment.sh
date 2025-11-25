#! /bin/bash

# OceanAI Document Generation Platform - Deployment Script
# Similar to Unthinkable deployment

echo "🌊 Starting OceanAI Deployment..."

# Update system
sudo apt update -y
sudo apt upgrade -y

# Install Python 3.11
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update -y
sudo apt install -y python3.11 python3.11-venv python3.11-dev nodejs npm nginx certbot python3-certbot-nginx

# Create virtual environment
python3.11 -m venv ~/oceanai/venv
source ~/oceanai/venv/bin/activate

# Install Python dependencies
pip install -r ~/oceanai/OceanAI/backend/requirements.txt

#App server
pip install gunicorn

# Build frontend
cd ~/oceanai/OceanAI/project/frontend/
npm install
npm run build

# Return to base directory
cd ~/oceanai

# Create Nginx configuration

# -- etc , bin , var  System-Level
nginx_file_path="/etc/nginx/sites-available/oceanai.conf"
sudo bash -c "cat > $nginx_file_path << 'EOF'
server {
    listen 80;
    server_name codemos-services.co.in www.codemos-services.co.in;
    
    # Frontend - Serve React build
    root /home/ubuntu/oceanai/OceanAI/project/frontend/dist;
    index index.html;
    
    location / {
        try_files \$uri \$uri/ @backend; 
    }
    
    # Backend API
    location @backend {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
    }
    
    # API routes (explicit)
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # API docs
    location /docs {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
    }
    
    location /openapi.json {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
    }
    
    access_log /var/log/nginx/oceanai_access.log;
    error_log /var/log/nginx/oceanai_error.log;
}
EOF"

# Enable site

# ln = link , -sf forceful link
sudo ln -sf /etc/nginx/sites-available/oceanai.conf /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx

# Setup SSL with Certbot
sudo rm -f /etc/letsencrypt/ssl-dhparams.pem
sudo certbot --nginx -d codemos-services.co.in -d www.codemos-services.co.in --non-interactive --agree-tos -m admin@codemos-services.co.in --redirect

# Enable Nginx on boot
sudo systemctl enable nginx

# Create log directory for Gunicorn
sudo mkdir -p /var/log/gunicorn
sudo chown -R ubuntu:www-data /var/log/gunicorn
sudo chmod -R 775 /var/log/gunicorn

# Create Gunicorn systemd service
gunicorn_file_path="/etc/systemd/system/oceanai-gunicorn.service"
sudo bash -c "cat > $gunicorn_file_path << 'EOF'
[Unit]
Description=OceanAI Gunicorn service for FastAPI
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/oceanai/OceanAI/backend
Environment="PATH=/home/ubuntu/oceanai/venv/bin"
ExecStart=/home/ubuntu/oceanai/venv/bin/gunicorn main:app \
  -k uvicorn.workers.UvicornWorker \
  --bind 127.0.0.1:8000 \
  --workers 4 --timeout 120 \
  --access-logfile /var/log/gunicorn/oceanai_access.log \
  --error-logfile /var/log/gunicorn/oceanai_error.log
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF"

# Reload systemd
sudo systemctl daemon-reload

# Enable and start Gunicorn
sudo systemctl enable oceanai-gunicorn
sudo systemctl start oceanai-gunicorn

# Set permissions for frontend
sudo chown -R ubuntu:www-data /home/ubuntu/oceanai/OceanAI/project/frontend/dist
sudo chmod -R 755 /home/ubuntu/oceanai/OceanAI/project/frontend/dist
sudo chmod 755 /home/ubuntu ~/oceanai ~/oceanai/OceanAI ~/oceanai/OceanAI/project/frontend

# Restart services
sudo systemctl restart nginx
sudo systemctl restart oceanai-gunicorn

echo "✅ OceanAI Deployment Complete!"
echo "🌐 Access at: https://codemos-services.co.in"
echo "📚 API Docs: https://codemos-services.co.in/docs"

# Check status
echo ""
echo "📊 Service Status:"
sudo systemctl status oceanai-gunicorn --no-pager
sudo systemctl status nginx --no-pager
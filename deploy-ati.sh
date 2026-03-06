#!/bin/bash

# ATI API Deployment Script
# Deploys to 65.0.65.2 (proxied through ati.13.201.80.220.nip.io/api)

set -e  # Exit on any error

# Configuration
SERVER_IP="65.0.65.2"
SERVER_USER="ubuntu"
PEM_FILE="./agri-chat.pem"
REMOTE_PATH="/home/ubuntu/oan-ai-api"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}   ATI API Deployment Script${NC}"
echo -e "${YELLOW}========================================${NC}"

# Check if PEM file exists
if [ ! -f "$PEM_FILE" ]; then
    echo -e "${RED}Error: PEM file not found at $PEM_FILE${NC}"
    exit 1
fi

# Step 1: Sync code (excluding venv, __pycache__, .git)
echo -e "\n${GREEN}[1/4] Syncing code to server...${NC}"
rsync -avz --progress \
    --exclude 'env/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude '.env' \
    --exclude '*.log' \
    --exclude '.pytest_cache/' \
    --exclude '.mypy_cache/' \
    -e "ssh -i $PEM_FILE" \
    ./ "$SERVER_USER@$SERVER_IP:$REMOTE_PATH/"

# Step 2: Install/update dependencies
echo -e "\n${GREEN}[2/4] Installing dependencies...${NC}"
ssh -i "$PEM_FILE" "$SERVER_USER@$SERVER_IP" \
    "cd $REMOTE_PATH && source env/bin/activate && pip install -r requirements.txt --quiet"

# Step 3: Stop existing server
echo -e "\n${GREEN}[3/4] Stopping existing server...${NC}"
ssh -i "$PEM_FILE" "$SERVER_USER@$SERVER_IP" \
    "pkill -f 'uvicorn main:app' || true"

# Step 4: Start server
echo -e "\n${GREEN}[4/4] Starting server...${NC}"
ssh -i "$PEM_FILE" "$SERVER_USER@$SERVER_IP" \
    "cd $REMOTE_PATH && source env/bin/activate && nohup uvicorn main:app --host 0.0.0.0 --port 8000 > /tmp/api.log 2>&1 &"

# Wait and verify
sleep 3
echo -e "\n${GREEN}Verifying deployment...${NC}"
ssh -i "$PEM_FILE" "$SERVER_USER@$SERVER_IP" \
    "ps aux | grep 'uvicorn main:app' | grep -v grep"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}   Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "API URL: https://ati.13.201.80.220.nip.io/api"
echo -e "Logs: ssh -i $PEM_FILE $SERVER_USER@$SERVER_IP 'tail -f /tmp/api.log'"

FROM python:3.11-slim

WORKDIR /app

# System deps (optional, keep minimal)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend code
COPY backend ./backend

# Copy start script
COPY scripts/start_server.sh ./scripts/start_server.sh
RUN chmod +x ./scripts/start_server.sh

ENV PORT=8000

EXPOSE 8000

CMD ["bash","-lc","./scripts/start_server.sh"]


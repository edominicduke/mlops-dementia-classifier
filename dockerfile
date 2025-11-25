FROM python:3.11

# Set working directory
WORKDIR /app

# Install system dependencies
# Created with ChatGPT 5.1 at 11/24/25 at 10:13pm for fixing errors with dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    build-essential && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Copy requirements files
COPY requirements-api.txt .

# Install all dependencies
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy entire project
COPY app/ ./app/
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/
COPY outputs/ ./outputs/
COPY data/ ./data/
COPY entrypoint.sh ./entrypoint.sh

RUN chmod +x /app/entrypoint.sh

# Expose port for API
ENV PORT=8080
EXPOSE 8080

# Use training/serving entrypoint router
# Entrypoint information was created via ChatGPT 5.1 11/24/25 at 10:10pm
ENTRYPOINT ["app/entrypoint.sh"]

# Default command
CMD ["serve"]
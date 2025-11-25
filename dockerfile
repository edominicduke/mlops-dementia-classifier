FROM python:3.11

# Set working directory
WORKDIR /app

# Install system dependencies (optional but safe)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    openssl \
    curl \
    gnupg \
    build-essential \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt .
COPY requirements-api.txt .

# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy entire project
COPY . .

# Expose port (Hugging Face, Docker, Cloud Run all use $PORT)
ENV PORT=8080
EXPOSE 8080

# Command to run your FastAPI app (YOUR API FILE IS IN ./app/app.py)

# Include entry points for serving/training (differentiate through POST /predict and POST /train)
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]
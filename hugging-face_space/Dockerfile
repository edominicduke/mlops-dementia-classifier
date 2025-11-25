# Use lightweight Python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only requirements first (better layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


COPY app.py .

# Hugging Face Spaces require port 7860
ENV PORT=7860
EXPOSE 7860

# Run FastAPI through uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
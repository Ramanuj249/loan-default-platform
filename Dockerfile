# Step 1 - Base image
FROM python:3.11-slim

# Step 2 - Set working directory inside container
WORKDIR /app

# Step 3 - Copy requirements first (for caching)
COPY requirements.txt .

# Step 4 - Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5 - Copy source code
COPY . .

# Step 6 - Expose port
EXPOSE 8000

# Step 7 - Run FastAPI using dynamic port from Render
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
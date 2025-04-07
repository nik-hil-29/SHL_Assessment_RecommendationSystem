FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    gnupg \
    curl \
    unzip \
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy setup script and requirements
COPY setup.sh requirements.txt ./

# Make setup script executable
RUN chmod +x setup.sh

# Run setup script and install requirements
RUN ./setup.sh && pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN playwright install

# Copy source code
COPY . .

# Make sure the data and database directories exist
RUN mkdir -p FinalDataSource
RUN mkdir -p chroma_db
RUN mkdir -p evaluation

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose port
EXPOSE 8000

# Command to run the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# Use official Python 3.13 image
FROM python:3.13-slim AS base

# Install system dependencies for llama-cpp and others
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (package manager)
RUN pip install uv

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY . .

# Install dependencies using uv
RUN uv pip install --no-cache --system .

EXPOSE 8080

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

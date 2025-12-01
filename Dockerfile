FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies for llama-cpp and others
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY . .

# Install dependencies using uv
RUN uv sync --frozen

# Run the FastAPI app with uvicorn
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

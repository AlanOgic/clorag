# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install dependencies
RUN uv sync --frozen --no-dev

# Production stage
FROM python:3.12-slim

WORKDIR /app

# Create non-root user
RUN groupadd --gid 1001 clorag && \
    useradd --uid 1001 --gid clorag --no-create-home clorag

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ ./src/

# Create data directory with correct ownership
RUN mkdir -p /app/data && chown -R clorag:clorag /app

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

# Run as non-root user
USER clorag

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "-m", "uvicorn", "clorag.web:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips=*"]

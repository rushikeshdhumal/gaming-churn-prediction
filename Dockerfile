# Gaming Player Behavior Analysis & Churn Prediction
# Multi-stage Dockerfile for containerized deployment
#
# Author: Rushikesh Dhumal
# Email: r.dhumal@rutgers.edu
#
# Build: docker build -t gaming-churn-prediction .
# Run:   docker run -it --rm -v $(pwd)/data:/app/data gaming-churn-prediction

# =============================================================================
# Stage 1: Base Environment Setup
# =============================================================================
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    make \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage 2: Dependencies Installation
# =============================================================================
FROM base as dependencies

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# =============================================================================
# Stage 3: Application Build
# =============================================================================
FROM dependencies as builder

# Copy source code
COPY . .

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data/{raw,processed,external} \
             models \
             reports/figures \
             logs \
             scripts/outputs \
             tests \
             docs

# Create .gitkeep files for empty directories
RUN touch data/processed/.gitkeep \
          data/external/.gitkeep \
          models/.gitkeep \
          reports/figures/.gitkeep \
          logs/.gitkeep \
          scripts/outputs/.gitkeep \
          tests/.gitkeep \
          docs/.gitkeep

# =============================================================================
# Stage 4: Production Image
# =============================================================================
FROM python:3.9-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PATH="/opt/venv/bin:$PATH" \
    ENVIRONMENT=production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code and directory structure
COPY --from=builder --chown=appuser:appuser /app .

# Create data volume mount point
VOLUME ["/app/data", "/app/models", "/app/logs"]

# Expose port (if needed for API)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import src.utils.config; print('Health check passed')" || exit 1

# Switch to non-root user
USER appuser

# Default command
CMD ["python", "scripts/run_complete_analysis.py", "--environment", "production"]

# =============================================================================
# Development Stage (for development workflow)
# =============================================================================
FROM builder as development

# Install development dependencies
RUN pip install -e .[dev]

# Install additional development tools
RUN pip install jupyter notebook ipykernel

# Set development environment
ENV ENVIRONMENT=development

# Expose Jupyter port
EXPOSE 8888

# Create Jupyter config
RUN mkdir -p /home/appuser/.jupyter
USER appuser

# Development command (Jupyter notebook)
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# =============================================================================
# Docker Compose Support
# =============================================================================

# Labels for metadata
LABEL maintainer="r.dhumal@rutgers.edu" \
      version="1.0.0" \
      description="Gaming Player Behavior Analysis & Churn Prediction" \
      license="MIT" \
      repository="https://github.com/rushikeshdhumal/gaming-churn-prediction"
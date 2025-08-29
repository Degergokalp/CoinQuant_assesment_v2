
# CoinQuant assessment dev image
FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Optional: Rust toolchain (for some NautilusTrader wheels)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Python deps
RUN pip install --no-cache-dir \
    nautilus-trader \
    openai jsonschema pandas matplotlib requests fsspec[http]

WORKDIR /app

CMD ["bash"]

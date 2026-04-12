FROM python:3.11-slim

WORKDIR /app

# Install graphviz system package for rendering.
RUN apt-get update && \
    apt-get install -y --no-install-recommends graphviz && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
COPY examples/ examples/
COPY README.md .
COPY LICENSE .

RUN pip install --no-cache-dir -e ".[all]"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/viz/app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0"]

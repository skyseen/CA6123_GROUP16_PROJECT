# HDB Resale Research Advisor: Streamlit UI, CLI, tests and evaluation in one image.
FROM python:3.12.8-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HDB_DATA_DIR=/app/data

WORKDIR /app

# Dependencies first so code edits do not invalidate the (slow) install layer.
COPY requirements.lock requirements-app.txt ./
RUN pip install -r requirements.lock -r requirements-app.txt

COPY hdb_advisor ./hdb_advisor
COPY scripts ./scripts
COPY evals ./evals
COPY config ./config
COPY examples ./examples
COPY .streamlit ./.streamlit
COPY streamlit_app.py walkthrough.ipynb README.md PHASE_SUMMARY.md ./

# Run as an unprivileged user. data/ and logs/ are mounted from the host by docker-compose.
RUN useradd --create-home --uid 1000 advisor && mkdir -p data logs && chown -R advisor /app
USER advisor

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]

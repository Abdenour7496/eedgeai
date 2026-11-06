FROM zepai/graphiti:0.31.2

USER root
RUN uv pip install --python /app/.venv/bin/python "graphiti-core[falkordb]==0.31.2" "prometheus-client>=0.21,<1"

COPY app.py /app/falkordb_app.py

# The upstream image runs as the unprivileged "app" user; drop back to it
# after installing extras instead of running the API as root.
USER app

CMD ["/app/.venv/bin/uvicorn", "falkordb_app:app", "--host", "0.0.0.0", "--port", "8000"]

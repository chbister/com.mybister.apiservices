#!/bin/sh
set -e

HOST="${UVICORN_HOST:-0.0.0.0}"
PORT="${UVICORN_PORT:-8000}"
LOG_LEVEL="${UVICORN_LOG_LEVEL:-info}"
WORKERS="${UVICORN_WORKERS:-1}"
RELOAD="${UVICORN_RELOAD:-0}"

ARGS="--host ${HOST} --port ${PORT} --log-level ${LOG_LEVEL}"

# In Prod i.d.R. mehrere Workers; in Dev meist 1 Worker + reload
if [ "${RELOAD}" = "1" ]; then
  ARGS="${ARGS} --reload"
else
  # Workers nur ohne reload sinnvoll
  if [ "${WORKERS}" != "1" ]; then
    ARGS="${ARGS} --workers ${WORKERS}"
  fi
fi

exec uvicorn app.main:app ${ARGS}

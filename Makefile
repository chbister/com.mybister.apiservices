.PHONY: help up-prod up-dev down-prod down-dev up-careergraph-prod up-careergraph-dev down-careergraph-prod down-careergraph-dev local-asr local-diarization local-debug-processor

PROFILE ?= dev
ACTION ?= up

# Help target
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  up-prod       - Start services in production profile"
	@echo "  up-dev        - Start services in development profile"
	@echo "  up-kickfriends-prod - Start services in kickfriends production profile"
	@echo "  up-kickfriends-dev  - Start services in kickfriends development profile"
	@echo "  up-careergraph-prod - Start CareerGraph services (ocr, ocr-worker, resume) in production"
	@echo "  up-careergraph-dev  - Start CareerGraph services (ocr-dev, ocr-worker-dev, resume-dev) for development"
	@echo "  local-asr           - Run ASR service locally (GPU endpoint /v1/asr/gpu, port 8100)"
	@echo "  local-diarization   - Run diarization service locally (/v1/diarize, port 8101)"
	@echo "  local-debug-processor - Run debug-processor service locally (/v1/transcript, port 8102)"
	@echo "  down-prod     - Stop services in production profile"
	@echo "  down-dev      - Stop services in development profile"
	@echo "  down-kickfriends-prod - Stop services in kickfriends production profile"
	@echo "  down-kickfriends-dev  - Stop services in kickfriends development profile"
	@echo "  down-careergraph-prod - Stop CareerGraph production services"
	@echo "  down-careergraph-dev  - Stop CareerGraph development services"
	@echo ""
	@echo "Or use: make compose PROFILE=<prod|dev|kickfriends-prod|kickfriends-dev|careergraph-prod|careergraph-dev> ACTION=<up|down>"

# Generic compose target
compose:
	@if [ "$(PROFILE)" != "prod" ] && [ "$(PROFILE)" != "dev" ] && [ "$(PROFILE)" != "kickfriends-prod" ] && [ "$(PROFILE)" != "kickfriends-dev" ] && [ "$(PROFILE)" != "careergraph-prod" ] && [ "$(PROFILE)" != "careergraph-dev" ]; then \
		echo "Error: Invalid profile. Use 'prod', 'dev', 'kickfriends-prod', 'kickfriends-dev', 'careergraph-prod' or 'careergraph-dev'."; \
		exit 1; \
	fi
	@if [ "$(ACTION)" != "up" ] && [ "$(ACTION)" != "down" ]; then \
		echo "Error: Invalid action. Use 'up' or 'down'."; \
		exit 1; \
	fi
	@if [ "$(ACTION)" = "up" ]; then \
		if [ "$(PROFILE)" = "prod" ]; then \
			docker compose --profile $(PROFILE) up -d --scale barcode=2 --scale metadata=2; \
		else \
			docker compose --profile $(PROFILE) up -d; \
		fi \
	else \
		docker compose --profile $(PROFILE) down; \
	fi

# ---------------------------------------------------------------------------
# Lokale Dev-Server (ohne Docker, damit Python die Apple-GPU nutzen kann)
# In eigenen Terminals starten; Ports kollidieren nicht mit Traefik (80/443).
# ---------------------------------------------------------------------------

local-asr:
	cd asr-service && ASR_JOBS_DIR=/tmp/asr-jobs ../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8100

local-diarization:
	cd diarization-service && DIAR_JOBS_DIR=/tmp/diar-jobs HF_TOKEN="$$(grep '^HF_TOKEN=' ../.env | cut -d= -f2-)" ../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8101

local-debug-processor:
	cd processor-service && ../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8102

# Convenience targets
up-prod:
	$(MAKE) compose PROFILE=prod ACTION=up

up-dev:
	docker network inspect dev-shared-network >/dev/null 2>&1 || docker network create dev-shared-network
	$(MAKE) compose PROFILE=dev ACTION=up

down-prod:
	$(MAKE) compose PROFILE=prod ACTION=down

down-dev:
	$(MAKE) compose PROFILE=dev ACTION=down

up-kickfriends-prod:
	$(MAKE) compose PROFILE=kickfriends-prod ACTION=up

down-kickfriends-prod:
	$(MAKE) compose PROFILE=kickfriends-prod ACTION=down

up-kickfriends-dev:
	docker network inspect dev-shared-network >/dev/null 2>&1 || docker network create dev-shared-network
	$(MAKE) compose PROFILE=kickfriends-dev ACTION=up

down-kickfriends-dev:
	$(MAKE) compose PROFILE=kickfriends-dev ACTION=down

up-careergraph-prod:
	$(MAKE) compose PROFILE=careergraph-prod ACTION=up

down-careergraph-prod:
	$(MAKE) compose PROFILE=careergraph-prod ACTION=down

up-careergraph-dev:
	docker network inspect dev-shared-network >/dev/null 2>&1 || docker network create dev-shared-network
	$(MAKE) compose PROFILE=careergraph-dev ACTION=up

down-careergraph-dev:
	$(MAKE) compose PROFILE=careergraph-dev ACTION=down

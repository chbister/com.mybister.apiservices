.PHONY: help up-prod up-dev down-prod down-dev

PROFILE ?= dev
ACTION ?= up

# Help target
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  up-prod       - Start services in production profile"
	@echo "  up-dev        - Start services in development profile"
	@echo "  down-prod     - Stop services in production profile"
	@echo "  down-dev      - Stop services in development profile"
	@echo ""
	@echo "Or use: make compose PROFILE=<prod|dev> ACTION=<up|down>"

# Generic compose target
compose:
	@if [ "$(PROFILE)" != "prod" ] && [ "$(PROFILE)" != "dev" ]; then \
		echo "Error: Invalid profile. Use 'prod' or 'dev'."; \
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

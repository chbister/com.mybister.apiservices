# Testing Documentation

This document describes how to execute tests locally and how the automated CI workflow works.

## Local Testing

To run the integration tests locally, you need `docker`, `docker-compose`, and `python3` with `pytest` installed.

### 1. Start Services

The tests interact with the services running in Docker. Use the `dev` profile to build and start them:

```bash
make up-dev
```

This will:
1. Create the `dev-shared-network` if it doesn't exist.
2. Build and start all services using the `dev` profile.

### 2. Prepare Python Environment

Install the required test dependencies:

```bash
pip install pytest httpx qrcode pillow
```

### 3. Run Tests

Execute `pytest` from the project root:

```bash
pytest tests/
```

By default, the tests assume services are available at `http://localhost:80` (through Traefik). You can override this with the `TEST_BASE_URL` environment variable:

```bash
TEST_BASE_URL=http://localhost:8080 pytest tests/
```

## How CI Works

The GitHub Actions workflow (`.github/workflows/ci.yml`) performs the following steps on every pull request and push to the `main` branch:

1.  **Build**: Builds all Docker images using `docker compose --profile dev build`.
2.  **Startup**: Starts services using `docker compose --profile dev up -d`.
3.  **Wait**: Waits for services to initialize and register with Traefik.
4.  **Test**: Runs the `pytest` suite against the running containers.
5.  **Logs**: If any step fails, it collects and displays container logs for troubleshooting.
6.  **Cleanup**: Shuts down and removes all containers and volumes.

## Adding New Service Tests

When adding a new service, please follow these steps to ensure it's covered by tests:

1.  **Health Check**: Add the service name to the `SERVICES` list in `tests/test_services.py` if it exposes a `/health` endpoint.
2.  **Integration/E2E Test**: Add a new test function in `tests/test_services.py` that validates the service's core functionality via its public API.
3.  **Test Data**: If your test requires specific data (audio, images, etc.), place it in `tests/data/`. Keep files small and representative.

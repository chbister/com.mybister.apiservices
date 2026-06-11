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
pytest -s tests/
```

The `-s` flag is recommended to see the readiness progress of the services.

By default, the tests assume services are available at `http://localhost:80` (through Traefik). You can override this with the following environment variables:

- `TEST_BASE_URL`: Base URL of the services (default: `http://localhost`)
- `SERVICE_STARTUP_TIMEOUT`: Maximum time in seconds to wait for services (default: `1200`)
- `SERVICE_POLL_INTERVAL`: Time in seconds between readiness checks (default: `5`)

Example:

```bash
TEST_BASE_URL=http://localhost:8080 SERVICE_STARTUP_TIMEOUT=600 pytest -s tests/
```

## How CI Works

The GitHub Actions workflow (`.github/workflows/ci.yml`) performs the following steps on every pull request and push to the `main` branch:

1.  **Build**: Builds all Docker images using `docker compose --profile dev build`.
2.  **Startup**: Starts services using `docker compose --profile dev up -d`.
3.  **Readiness**: Pytest automatically waits for all services to be healthy via a session-scoped fixture before running any tests.
4.  **Test**: Runs the `pytest` suite against the running containers.
5.  **Logs**: If any step fails, it collects and displays container logs for troubleshooting.
6.  **Cleanup**: Shuts down and removes all containers and volumes.

## Adding New Service Tests

When adding a new service, please follow these steps to ensure it's covered by tests:

1.  **Health Check**: Add the service name to the `SERVICES` list in `tests/conftest.py` if it exposes a `/health` endpoint.
2.  **Integration/E2E Test**: Add a new test function in `tests/test_services.py` that validates the service's core functionality via its public API. Also add it to the `SERVICES` list in `tests/test_services.py` if you want it parameterized in the health check test.
3.  **Test Data**: If your test requires specific data (audio, images, etc.), place it in `tests/data/`. Keep files small and representative.

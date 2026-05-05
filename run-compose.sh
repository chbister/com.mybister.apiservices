#!/bin/bash

# Check arguments
if [ $# -lt 2 ]; then
  echo "Usage: $0 [prod|dev] [up|down]"
  exit 1
fi

PROFILE="$1"
ACTION="$2"

# Validate profile
if [[ "$PROFILE" != "prod" && "$PROFILE" != "dev" ]]; then
  echo "Error: Invalid profile. Use 'prod' or 'dev'."
  exit 1
fi

# Validate action
if [[ "$ACTION" != "up" && "$ACTION" != "down" ]]; then
  echo "Error: Invalid action. Use 'up' or 'down'."
  exit 1
fi

# Execute command
if [ "$ACTION" == "up" ]; then
  docker compose --profile "$PROFILE" up -d --scale barcode=2 --scale metadata=2
else
  docker compose --profile "$PROFILE" down
fi

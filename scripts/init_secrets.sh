#!/usr/bin/env bash
# Initialize Docker Secrets directory from existing .env file
# Run once on the server to migrate from .env to Docker Secrets
#
# Usage: ./scripts/init_secrets.sh [env_file]
#   env_file: path to .env file (default: .env)

set -euo pipefail

ENV_FILE="${1:-.env}"
SECRETS_DIR="./secrets"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found"
    exit 1
fi

# Secret env vars to extract
SECRET_KEYS=(
    "ANTHROPIC_API_KEY"
    "VOYAGE_API_KEY"
    "QDRANT_API_KEY"
    "ADMIN_PASSWORD"
    "NEO4J_PASSWORD"
    "OPENAI_COMPAT_API_KEY"
    "ODOO_MCP_API_KEY"
)

# Create secrets directory
mkdir -p "$SECRETS_DIR"
chmod 700 "$SECRETS_DIR"

echo "Extracting secrets from $ENV_FILE into $SECRETS_DIR/"

for key in "${SECRET_KEYS[@]}"; do
    # Extract value from .env (handles quotes and inline comments)
    value=$(grep -E "^${key}=" "$ENV_FILE" | head -1 | sed 's/^[^=]*=//' | sed 's/#.*//' | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//' | sed 's/^["'\'']//' | sed 's/["'\'']$//')

    # Convert KEY_NAME to lowercase file name (e.g. ANTHROPIC_API_KEY -> anthropic_api_key)
    filename=$(echo "$key" | tr '[:upper:]' '[:lower:]')

    if [ -n "$value" ]; then
        printf '%s' "$value" > "$SECRETS_DIR/$filename"
        chmod 600 "$SECRETS_DIR/$filename"
        echo "  Created: $SECRETS_DIR/$filename"
    else
        echo "  Skipped: $key (not found or empty in $ENV_FILE)"
    fi
done

echo ""
echo "Done. Secrets stored in $SECRETS_DIR/ with mode 600."
echo ""
echo "Next steps:"
echo "  1. Verify secrets: ls -la $SECRETS_DIR/"
echo "  2. Rebuild containers: docker compose build && docker compose up -d"
echo "  3. Verify health: docker compose ps"
echo "  4. Once confirmed working, remove secrets from .env"

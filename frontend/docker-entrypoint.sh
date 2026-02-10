#!/bin/sh
# Generate runtime config from environment variables.
# This allows a single Docker image to work across all environments
# (dev, staging, production) by injecting config at container start.
cat > /usr/share/nginx/html/config.js <<EOF
window.__CONFIG__ = {
  API_URL: "${API_URL:-http://localhost:8000}",
  API_KEY: "${VITE_API_KEY:-}"
};
EOF
exec nginx -g 'daemon off;'

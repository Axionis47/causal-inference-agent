# Frontend Dockerfile for Causal Inference Orchestrator

# =============================================================================
# Development stage - with hot reload
# =============================================================================
FROM node:20-alpine AS development

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install all dependencies including devDependencies
RUN npm install

# Copy source code
COPY . .

# Expose Vite dev server port
EXPOSE 5173

# Run dev server
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# =============================================================================
# Builder stage - compile production assets
# =============================================================================
FROM node:20-alpine AS builder

WORKDIR /app

# Accept build-time variables
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

# Copy package files
COPY package*.json ./

# Install dependencies (production only for smaller node_modules)
RUN npm ci

# Copy source code
COPY . .

# Build the app
RUN npm run build

# =============================================================================
# Production stage - nginx serving static files
# =============================================================================
FROM nginx:alpine AS production

RUN apk add --no-cache curl

# Copy nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy runtime config entrypoint
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Expose port (Cloud Run uses 8080)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Use entrypoint for runtime config injection
ENTRYPOINT ["/docker-entrypoint.sh"]

# =============================================================================
# Default to production
# =============================================================================
FROM production

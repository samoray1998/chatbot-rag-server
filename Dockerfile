# FROM node:18-alpine AS builder
# WORKDIR /app
# COPY package*.json ./
# RUN npm ci
# COPY . .
# RUN npm run build

# FROM node:18-alpine
# WORKDIR /app
# COPY --from=builder /app/node_modules ./node_modules
# COPY --from=builder /app/dist ./dist
# COPY package.json .

# # Only install JS dependencies (no Python needed)
# USER node
# EXPOSE 3000
# CMD ["node", "dist/index.js"]


# Base image for both production and development
FROM node:18-alpine AS base
WORKDIR /app
COPY package*.json ./

# Development stage
FROM base AS dev
# Install additional tools for development if needed
# RUN apk add --no-cache bash curl vim
RUN npm install
COPY . .
# Use nodemon for live reloading in development
RUN npm install -g nodemon
EXPOSE 3000
# Use nodemon to watch for changes (make sure it's installed as a dev dependency)
CMD ["npm", "run", "dev:docker"]

# Builder stage for production
FROM base AS builder
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM node:18-alpine AS production
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY package.json .
USER node
EXPOSE 3000
CMD ["node", "dist/index.js"]

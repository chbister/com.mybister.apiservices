# ics-service – ICS-Kalendergenerator (Node.js, analog zu ocrpdf-service).
FROM node:22-alpine

WORKDIR /app

# Lockfile zuerst – Layer-Cache bleibt bei Code-Änderungen gültig.
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY . .

EXPOSE 3000

CMD ["node", "server.js"]

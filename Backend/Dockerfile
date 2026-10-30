FROM rust:1-slim-trixie AS build

WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Cache des dépendances
COPY Backend/Cargo.toml Backend/Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs \
    && cargo build --release && rm -rf src

COPY Backend/src ./src
RUN touch src/main.rs && cargo build --release

# ONNX Runtime est lié statiquement dans le binaire
FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /src/target/release/backend ./
COPY IA/ ./IA/

ENV S1_BASE_DIR=/app/IA/Sujet_1
ENV S2_BASE_DIR=/app/IA/Sujet_2
ENV S3_BASE_DIR=/app/IA/Sujet_3

EXPOSE 8000

CMD ["./backend"]

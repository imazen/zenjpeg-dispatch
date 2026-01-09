# Dockerfile for zenjpeg discover_heuristics benchmark tool
#
# This project depends on local workspace crates (mozjpeg-rs, jpegli-rs, etc.)
# so the Docker build must be run from the parent directory with all dependencies.
#
# Build from ~/work directory:
#   docker build -t discover-heuristics -f zenjpeg/Dockerfile .
#
# Run:
#   docker run --rm discover-heuristics --help
#   docker run --rm -v /path/to/corpus:/corpus -v /path/to/output:/output \
#     discover-heuristics --corpus /corpus --output /output

# Stage 1: Build
FROM rust:1.83-bookworm as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    pkg-config \
    libwebp-dev \
    nasm \
    && rm -rf /var/lib/apt/lists/*

# Note: For AVIF decoding, add: libdav1d-dev
# Then enable avif-native feature in Cargo.toml image dependency

WORKDIR /build

# Copy all workspace dependencies (build from parent directory)
COPY mozjpeg-rs /build/mozjpeg-rs
COPY jpegli-rs /build/jpegli-rs
COPY butteraugli /build/butteraugli
COPY codec-eval /build/codec-eval

# Copy zenjpeg project
COPY zenjpeg /build/zenjpeg

WORKDIR /build/zenjpeg

# Build release binary with WebP support
# Note: AVIF encoding works (ravif is pure Rust), but AVIF decoding requires libdav1d
RUN cargo build --release --features webp --example discover_heuristics

# Stage 2: Runtime
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libwebp7 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy binary
COPY --from=builder /build/zenjpeg/target/release/examples/discover_heuristics /app/discover_heuristics

# Set working directory for data
WORKDIR /data

# Default entrypoint
ENTRYPOINT ["/app/discover_heuristics"]

# Default command (show help)
CMD ["--help"]

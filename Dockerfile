FROM dailyco/pipecat-base:latest

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Copy only pyproject first (enables Docker layer caching)
COPY pyproject.toml ./

# Generate lockfile inside the build (works even if repo has no uv.lock)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv lock

# Install deps from the generated lockfile
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Copy app code
COPY ./bot.py ./bot.py

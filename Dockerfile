# Wingspan cloud trainer — Amazon Linux 2023, CPU-only torch.
#
# Build:  docker build -t wingspan-trainer .
# Run:    docker run --rm -v "$PWD/run.yaml:/config/run.yaml" wingspan-trainer
# (Training is CPU-only — see CLAUDE.md — so there is no CUDA base image.)
FROM amazonlinux:2023

# Python 3.12 + minimal build tools. git is best-effort: the status snapshot
# records a short SHA when a repo is present, but .git is excluded from the image
# (see .dockerignore), so in the container it simply resolves to null.
RUN dnf install -y python3.12 python3.12-pip git tar gzip \
    && dnf clean all \
    && rm -rf /var/cache/dnf

WORKDIR /app

# Install the CPU-only torch wheel FIRST so the project install below sees
# torch>=2.0 already satisfied and never pulls the multi-GB CUDA build.
RUN python3.12 -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Then the project itself (only src/ is packaged; see [tool.setuptools] config).
COPY pyproject.toml README.md ./
COPY src ./src
RUN python3.12 -m pip install --no-cache-dir .

# Run unprivileged; /work is the per-run scratch dir the run-file defaults to
# (the durable copy of every artifact lives in S3).
RUN useradd --create-home --uid 10001 flyway \
    && mkdir -p /work \
    && chown flyway:flyway /work
USER flyway
WORKDIR /work

ENTRYPOINT ["python3.12", "-m", "wingspan.cloud"]
CMD ["--config", "/config/run.yaml"]

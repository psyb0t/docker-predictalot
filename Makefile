PORT ?= 8080

DEV_IMAGE := psyb0t/predictalot-dev:latest
CPU_IMAGE := psyb0t/predictalot:local
CUDA_IMAGE := psyb0t/predictalot:local-cuda

PYPROJECT := pyproject.toml
BUMP_HOST := bash scripts/bump_exclude_newer.sh $(PYPROJECT)

UID := $(shell id -u)
GID := $(shell id -g)
DOCKER_SOCK := /var/run/docker.sock
DOCKER_GID := $(shell stat -c '%g' $(DOCKER_SOCK) 2>/dev/null || echo 0)

# Sandboxed dev container — all dev-side commands run inside this so the host
# stays clean. The full dev env is baked into /opt/venv at image build time, so
# there's no .venv on the host bind-mount. Lockfile changes → next `dev-image`
# build picks them up via docker layer cache invalidation on the COPY step.
DEV_RUN := docker run --rm \
	-u $(UID):$(GID) \
	-e HOME=/tmp \
	-v $(PWD):/work \
	-w /work \
	$(DEV_IMAGE)

DEV_RUN_TTY := docker run --rm -it \
	-u $(UID):$(GID) \
	-e HOME=/tmp \
	-v $(PWD):/work \
	-w /work \
	$(DEV_IMAGE)

.PHONY: help dev-image shell \
        pkg-lock pkg-upgrade pkg-add pkg-remove pkg-update \
        deps-lock \
        build build-cuda build-all \
        run run-cuda \
        test test-unit test-integration \
        lint format check clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# -----------------------------------------------------------------------------
# Dev container — every other target depends on this.
# -----------------------------------------------------------------------------

dev-image: ## Build/refresh the sandboxed dev image
	docker build -f Dockerfile.dev -t $(DEV_IMAGE) .

shell: dev-image ## Drop into a shell inside the dev container
	$(DEV_RUN_TTY) bash

# -----------------------------------------------------------------------------
# Lockfile mutations — only mutate pyproject.toml + uv.lock on the bind-mount.
# Every command bumps exclude-newer to today first so the supply-chain age
# gate is anchored to the moment of the change.
# -----------------------------------------------------------------------------

pkg-lock: dev-image ## Refresh uv.lock (honors exclude-newer)
	$(DEV_RUN) uv lock

pkg-upgrade: dev-image ## Bump exclude-newer + refresh lock with newest pins
	$(BUMP_HOST)
	$(DEV_RUN) uv lock --upgrade

pkg-add: dev-image ## Add a package (usage: make pkg-add PKG=name[==ver])
	@test -n "$(PKG)" || (echo "usage: make pkg-add PKG=name[==ver]" >&2; exit 1)
	$(BUMP_HOST)
	$(DEV_RUN) uv add --no-sync $(PKG)

pkg-remove: dev-image ## Remove a package (usage: make pkg-remove PKG=name)
	@test -n "$(PKG)" || (echo "usage: make pkg-remove PKG=name" >&2; exit 1)
	$(BUMP_HOST)
	$(DEV_RUN) uv remove --no-sync $(PKG)

pkg-update: dev-image ## Upgrade a package (usage: make pkg-update PKG=name)
	@test -n "$(PKG)" || (echo "usage: make pkg-update PKG=name" >&2; exit 1)
	$(BUMP_HOST)
	$(DEV_RUN) uv lock --upgrade-package $(PKG)

# Regenerate the hash-locked hash-locked ML deps (one per Docker variant).
# Run this any time scripts/deps.in changes. Bumps exclude-newer first
# so the supply-chain age gate is anchored to the moment of the change.
deps-lock: dev-image ## Recompile requirements-{cpu,cuda}.txt with hashes
	$(BUMP_HOST)
	$(DEV_RUN) bash scripts/compile-deps.sh

# -----------------------------------------------------------------------------
# Production image builds.
# -----------------------------------------------------------------------------

build: ## Build the CPU production image
	docker build -f Dockerfile -t $(CPU_IMAGE) .

build-cuda: ## Build the CUDA production image
	docker build -f Dockerfile.cuda -t $(CUDA_IMAGE) .

build-all: build build-cuda ## Build both production images

# -----------------------------------------------------------------------------
# Local run targets.
# -----------------------------------------------------------------------------

run: build ## Run CPU image locally (uses ~/.predictalot-models for cache)
	mkdir -p $$HOME/.predictalot-models
	docker run --rm -it \
		-v $$HOME/.predictalot-models:/models \
		-e PREDICTALOT_AUTH_TOKENS=devtoken \
		-e PREDICTALOT_DEVICE=cpu \
		-p $(PORT):8080 \
		$(CPU_IMAGE)

run-cuda: build-cuda ## Run CUDA image locally (requires --gpus all support)
	mkdir -p $$HOME/.predictalot-models
	docker run --rm -it --gpus all \
		-v $$HOME/.predictalot-models:/models \
		-e PREDICTALOT_AUTH_TOKENS=devtoken \
		-p $(PORT):8080 \
		$(CUDA_IMAGE)

# -----------------------------------------------------------------------------
# Test / lint / format — all inside the dev container.
# -----------------------------------------------------------------------------

test: test-unit ## Run all tests

test-unit: dev-image ## Run unit tests with stubbed backends
	$(DEV_RUN) pytest

# Integration tests run on the host (not in the dev container) because
# (a) they need to spawn sibling containers via the host docker daemon and
# (b) the DIND ceremony (--group-add docker, same-path bind-mount) buys
# nothing here — the host already has docker + uv, and uv keeps its own
# isolated .venv so the host isn't polluted.
#
# The fixture detects CUDA by inspecting `docker info` for the nvidia
# runtime — if present, Dockerfile.cuda is built and `--gpus all` is passed
# to the test container; otherwise Dockerfile. Models cache to
# tests/integration/.fixtures/models (gitignored).
test-integration: ## Build the image + run real-inference tests (uses CUDA if host has it)
	uv run --group dev pytest tests/integration -m integration

bench: ## Accuracy + latency benchmark on real public datasets (needs running container)
	uv run --group dev python benchs/run.py

lint: dev-image ## Lint python sources
	$(DEV_RUN) flake8 src tests
	$(DEV_RUN) mypy src

format: dev-image ## Format python sources
	$(DEV_RUN) isort src tests
	$(DEV_RUN) black src tests

check: lint test ## Lint + tests

clean: ## Remove build / cache artifacts (host-side)
	docker rmi $(CPU_IMAGE) $(CUDA_IMAGE) 2>/dev/null || true
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache .venv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

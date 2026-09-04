.PHONY: build test fmt lint clean help all fetch-model

help:
	@echo "AI Kit — Makefile targets"
	@echo ""
	@echo "  make build        Build release binary"
	@echo "  make test         Run all tests"
	@echo "  make fmt          Format code"
	@echo "  make lint         Run clippy linter"
	@echo "  make check        Run fmt + lint + test (pre-commit)"
	@echo "  make clean        Remove build artifacts"
	@echo "  make all          Build + test + fmt + lint"
	@echo "  make fetch-model  Download a small model for local --features candle"
	@echo "                    testing (POC only). MODEL=name to pick, default tinyllama."
	@echo ""

all: build test fmt lint

build:
	cargo build --release

test:
	cargo test

fetch-model:
	./scripts/fetch-model.sh $(MODEL)

fmt:
	cargo fmt

lint:
	cargo clippy --all-targets -- -D warnings

check: fmt lint test
	@echo "✓ All checks passed"

clean:
	cargo clean

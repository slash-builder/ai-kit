#!/usr/bin/env bash
# Fetch a small local dev model for AI Kit's `candle` feature — POC ONLY.
#
# Production models ship signed in the OS image (invariant #1/#2: no
# weights over network, models are immutable OS resources). This script is
# a deliberate, documented exception for local developer-machine testing:
# it downloads a small quantized GGUF model from Hugging Face so
# `ai-kit-cli infer` has something real to run against without BenixOS,
# Storage Kit, or an OS image. Never used in production, never wired into
# any boot path.
#
# Usage: ./scripts/fetch-model.sh [model-name]
#   model-name: one of the keys in the table below. Default: tinyllama.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL_NAME="${1:-tinyllama}"

# --- model table --------------------------------------------------------
# Plain-Llama-architecture models only for now: candle-transformers'
# quantized_llama path is the most mature/best-supported GGUF loader, which
# is why TinyLlama is the default rather than a Qwen/Mistral variant.
case "$MODEL_NAME" in
  tinyllama)
    HF_REPO="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    GGUF_FILE="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    TOKENIZER_REPO="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ARCHITECTURE="llama"
    CHAT_TEMPLATE="zephyr"
    QUANTIZATION="q4_k_m"
    PARAMETERS=1100000000
    CONTEXT_WINDOW=2048
    VRAM_MB=1024
    ;;
  *)
    echo "error: unknown model '$MODEL_NAME'. Known models: tinyllama" >&2
    exit 1
    ;;
esac

MODEL_DIR="dev-models/$MODEL_NAME"
MODEL_PATH="$MODEL_DIR/model.gguf"
TOKENIZER_PATH="$MODEL_DIR/tokenizer.json"

mkdir -p "$MODEL_DIR"

fetch() {
  local url="$1" dest="$2"
  if [ -f "$dest" ] && [ -s "$dest" ]; then
    echo "  already have $dest, skipping"
    return 0
  fi
  echo "  fetching $url"
  # Fail loudly on a bad status rather than silently writing a partial/error
  # body to disk. --fail-with-body would be nicer but isn't universally
  # available yet; -w lets us check the status explicitly instead.
  local tmp status
  tmp="$(mktemp)"
  status="$(curl -L -sS -o "$tmp" -w '%{http_code}' "$url")"
  if [ "$status" != "200" ]; then
    rm -f "$tmp"
    echo "error: fetch failed ($status) for $url" >&2
    exit 1
  fi
  mv "$tmp" "$dest"
}

echo "AI Kit dev model fetch: $MODEL_NAME"
fetch "https://huggingface.co/${HF_REPO}/resolve/main/${GGUF_FILE}" "$MODEL_PATH"
fetch "https://huggingface.co/${TOKENIZER_REPO}/resolve/main/tokenizer.json" "$TOKENIZER_PATH"

SIZE_BYTES="$(stat -f%z "$MODEL_PATH" 2>/dev/null || stat -c%s "$MODEL_PATH")"

echo "  building ai-kit-cli (hash subcommand only, no candle feature needed)"
cargo build --bin ai-kit-cli --quiet

HASH="$(./target/debug/ai-kit-cli hash "$MODEL_PATH")"

echo "  writing dev-models/catalog.yaml"
sed \
  -e "s|__MODEL_ID__|${MODEL_NAME}|g" \
  -e "s|__QUANTIZATION__|${QUANTIZATION}|g" \
  -e "s|__BLAKE3_HASH__|${HASH}|g" \
  -e "s|__SIZE_BYTES__|${SIZE_BYTES}|g" \
  -e "s|__MODEL_PATH__|file://${REPO_ROOT}/${MODEL_PATH}|g" \
  -e "s|__PARAMETERS__|${PARAMETERS}|g" \
  -e "s|__CONTEXT_WINDOW__|${CONTEXT_WINDOW}|g" \
  -e "s|__VRAM_MB__|${VRAM_MB}|g" \
  -e "s|__LATENCY_MS__|0|g" \
  -e "s|__TOKENIZER_PATH__|${REPO_ROOT}/${TOKENIZER_PATH}|g" \
  -e "s|__ARCHITECTURE__|${ARCHITECTURE}|g" \
  -e "s|__CHAT_TEMPLATE__|${CHAT_TEMPLATE}|g" \
  dev-models/catalog.yaml.tmpl > dev-models/catalog.yaml

cat <<'BANNER'

POC-ONLY: model fetched over network for local dev inference.
Production ships models signed in the OS image (invariant #2) — this
exception is documented in README.md ("Local Dev Inference (POC)") and
AGENT.md. Never used at runtime on a real device.

Next: cargo build --features candle
      cargo run --features candle --bin ai-kit-cli -- infer --model MODEL_NAME_PLACEHOLDER --prompt "Say hello"
BANNER
echo "(replace MODEL_NAME_PLACEHOLDER above with: $MODEL_NAME)"

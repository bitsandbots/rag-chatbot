#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
skip()  { echo -e "${YELLOW}[skip]${NC} $1 already exists"; }
error() { echo -e "${RED}[error]${NC} $1" >&2; exit 1; }

echo "============================================"
echo "  RAG Chatbot — Fresh Install"
echo "============================================"
echo

# --- Python 3.11+ check ---
if command -v python3 &>/dev/null; then
    PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    PYTHON_MAJOR="$(echo "$PYTHON_VERSION" | cut -d. -f1)"
    PYTHON_MINOR="$(echo "$PYTHON_VERSION" | cut -d. -f2)"
    if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]; }; then
        error "Python 3.11+ required, found Python ${PYTHON_VERSION}. Install a newer version first."
    fi
    info "Python ${PYTHON_VERSION} found"
else
    error "python3 not found. Install Python 3.11+ first."
fi

# --- Ollama check ---
if command -v ollama &>/dev/null; then
    info "Ollama found: $(ollama --version 2>&1 | head -1)"
else
    warn "Ollama not found."
    read -r -p "Install Ollama now? [y/N] " response
    if [[ "${response,,}" == "y" ]]; then
        info "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
    else
        error "Ollama is required. Install it from https://ollama.com and re-run this script."
    fi
fi

# --- Pull required models ---
info "Pulling nomic-embed-text..."
ollama pull nomic-embed-text

info "Pulling qwen2.5-coder:3b..."
ollama pull qwen2.5-coder:3b

# --- Create virtualenv ---
VENV="$PROJECT_DIR/venv"
if [ -d "$VENV" ]; then
    skip "venv"
else
    info "Creating virtual environment..."
    python3 -m venv "$VENV"
fi

# --- Install package ---
info "Installing package with dev dependencies..."
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -e "$PROJECT_DIR[dev]" build

# --- Copy .env ---
ENV_FILE="$PROJECT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    skip ".env"
else
    info "Creating .env from .env.example..."
    cp "$PROJECT_DIR/.env.example" "$ENV_FILE"
fi

# --- Create documents/ directory ---
DOCS_DIR="$PROJECT_DIR/documents"
if [ -d "$DOCS_DIR" ]; then
    skip "documents/"
else
    info "Creating documents/ directory..."
    mkdir -p "$DOCS_DIR"
fi

echo
echo "============================================"
echo "  Install complete!"
echo "============================================"
echo
echo "Next steps:"
echo "  1. Review .env and adjust settings if needed"
echo "  2. Add documents to documents/ for ingestion"
echo "  3. Run:  make run"
echo "  4. Open: http://localhost:5000"
echo

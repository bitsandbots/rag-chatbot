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
error() { echo -e "${RED}[error]${NC} $1" >&2; exit 1; }

echo "============================================"
echo "  RAG Chatbot — Release"
echo "============================================"
echo

# --- Version argument ---
VERSION="${1:-}"
if [ -z "$VERSION" ]; then
    error "Usage: $0 <version>  (e.g. $0 0.2.0)"
fi

# --- Validate semver X.Y.Z ---
if ! echo "$VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    error "Invalid version '${VERSION}'. Must be semver format X.Y.Z (e.g. 0.2.0)"
fi

info "Releasing version ${VERSION}"

# --- Require clean working tree ---
cd "$PROJECT_DIR"
if ! git diff --quiet || ! git diff --cached --quiet; then
    error "Working tree is not clean. Commit or stash changes before releasing."
fi

# --- Run tests ---
info "Running test suite..."
if [ -d "$PROJECT_DIR/venv" ]; then
    PYTEST="$PROJECT_DIR/venv/bin/pytest"
else
    PYTEST="pytest"
fi
"$PYTEST" -q || error "Tests failed. Fix before releasing."

# --- Run lint ---
info "Running lint..."
if [ -d "$PROJECT_DIR/venv" ]; then
    RUFF="$PROJECT_DIR/venv/bin/ruff"
else
    RUFF="ruff"
fi
"$RUFF" check src/ tests/ || error "Lint failed. Fix before releasing."

# --- Update version in __init__.py ---
INIT_FILE="$PROJECT_DIR/src/rag_chatbot/__init__.py"
if [ -f "$INIT_FILE" ]; then
    info "Updating version in src/rag_chatbot/__init__.py..."
    sed -i "s/__version__ = \".*\"/__version__ = \"${VERSION}\"/" "$INIT_FILE"
else
    warn "src/rag_chatbot/__init__.py not found — skipping __version__ update"
fi

# --- Update version in pyproject.toml ---
info "Updating version in pyproject.toml..."
sed -i "s/^version = \".*\"/version = \"${VERSION}\"/" "$PROJECT_DIR/pyproject.toml"

# --- Build the package ---
info "Building package..."
if [ -d "$PROJECT_DIR/venv" ]; then
    "$PROJECT_DIR/venv/bin/python" -m build "$PROJECT_DIR"
else
    python3 -m build "$PROJECT_DIR"
fi

# --- Commit and tag ---
info "Committing release..."
git add src/rag_chatbot/__init__.py pyproject.toml
git commit -m "chore(release): v${VERSION}"

info "Creating tag v${VERSION}..."
git tag "v${VERSION}"

echo
echo "============================================"
echo "  Release v${VERSION} ready!"
echo "============================================"
echo
echo "Next steps (run manually to publish):"
echo "  git push origin main"
echo "  git push origin v${VERSION}"
echo "  gh release create v${VERSION} dist/* \\"
echo "    --title \"v${VERSION}\" --generate-notes"
echo

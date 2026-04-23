#!/bin/bash
# download-icons.sh — Download Lucide icons relative to script location

# 1. Get the directory where this script actually lives
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ICONS_DIR="$PROJECT_ROOT/static/icons/lucide"

# 2. Create the directory
mkdir -p "$ICONS_DIR"

# 3. List of icons to fetch
ICONS=(
  "mic" "square" "send" "x-circle"
  "activity" "brain" "heart-pulse" "volume-2"
  "info" "settings" "user" "home"
  "smartphone" "wifi-off" "refresh-cw" "check"
)

BASE="https://unpkg.com/lucide@latest/icons"

echo "Project Root: $PROJECT_ROOT"
echo "Downloading ${#ICONS[@]} icons to $ICONS_DIR..."
echo "------------------------------------------------"

# 4. Loop and download
for icon in "${ICONS[@]}"; do
    # -s: silent (cleaner output)
    # -f: fail on 404
    # -L: follow redirects (required for unpkg)
    if curl -L "$BASE/$icon.svg" -o "$ICONS_DIR/$icon.svg"; then
        echo "  ✓ $icon.svg"
    else
        echo "  ✗ $icon.svg (failed)"
    fi
done

echo "------------------------------------------------"
echo "Done! All icons saved."

#!/bin/bash

set -e  # Exit immediately if a command fails
set -o pipefail

# CONFIGURATION
GITHUB_REPO="https://github.com/neuroptimiser/neuroptimiser.github.io.git"
BUILD_DIR="docs/build/html"
DEPLOY_DIR="/tmp/neuroptimiser-docs-deploy"

echo "🚀 Starting documentation deployment"

# Step 1 — Build Sphinx documentation locally
echo "🔧 Building documentation"
make -C docs html

# Step 2 — Prepare deployment directory
echo "📂 Preparing deploy folder"
rm -rf "$DEPLOY_DIR/"
git clone "$GITHUB_REPO" "$DEPLOY_DIR"

# Step 3 — Sync built HTML to deployment repo
echo "📄 Copying generated files"
rsync -av --delete "$BUILD_DIR/" "$DEPLOY_DIR/"

# Step 4 — Commit and push if changes exist
cd "$DEPLOY_DIR"

echo "✅ Changes detected, committing..."
git add --all
git commit -m "Update documentation: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
git push origin main
echo "🚀 Deployment successful!"

# Step 5 — Clean up
rm -rf "$DEPLOY_DIR"
echo "🧹 Cleanup complete"
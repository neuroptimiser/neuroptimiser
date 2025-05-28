#!/bin/bash

# Build docs
make -C docs html

# Clean and prepare temporary deploy folder
rm -rf /tmp/docs-deploy/*
git worktree add /tmp/docs-deploy gh-pages

# Copy built docs
rsync -av docs/build/html/ /tmp/docs-deploy/

# Commit and push
cd /tmp/docs-deploy
git add --all
git commit -m "Update docs $(date)"
git push origin gh-pages

# Cleanup
git worktree remove /tmp/docs-deploy
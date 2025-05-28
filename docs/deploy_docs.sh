#!/bin/bash

# Build docs
make -C docs html

# Switch to gh-pages branch
git worktree add /tmp/docs-deploy gh-pages

# Copy built docs
rm -rf /tmp/docs-deploy/*
cp -r docs/_build/html/* /tmp/docs-deploy/

# Commit and push
cd /tmp/docs-deploy
git add --all
git commit -m "Update docs $(date)"
git push origin gh-pages

# Cleanup
git worktree remove /tmp/docs-deploy
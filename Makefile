.PHONY: sync install editable test

## Sync the environment and reinstall editable package
sync:
	uv sync
	uv pip install -e .

## Install just the editable package
editable:
	uv pip install -e .

## Compile dependencies and lock
compile:
	uv pip compile pyproject.toml

## Run tests
test:
	pytest tests

## Clean up build artifacts
clean:
	find . -type d -name '__pycache__' -exec rm -r {} +
	rm -rf *.egg-info dist build
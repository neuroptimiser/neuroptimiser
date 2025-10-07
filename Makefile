.PHONY: sync install editable test compile clean build check upload testpypi

## Sync the environment and reinstall editable package
sync:
	uv sync --all-extras
	uv pip install -e ".[dev]"

## Install just the editable package
editable:
	uv pip install -e ".[dev]"

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

## Build the package (wheel + sdist)
build:
	python -m build

## Check built distributions
check:
	twine check dist/*

## Upload to PyPI
upload: build check
	twine upload dist/*

## Upload to TestPyPI
testpypi: build check
	twine upload --repository testpypi dist/*
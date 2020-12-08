.PHONY: quality style test

# Check that source code meets quality standards

quality:
	black --check --line-length 119 --target-version py36 src scripts tests
	isort --check-only --recursive src scripts tests
	flake8 src scripts tests

# Format source code automatically

style:
	black --line-length 119 --target-version py36 src scripts tests
	isort --recursive src scripts tests

# Run tests for the library

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

# Build or rebuild local library package

build:
	python setup.py install

rebuild:
	rm -rf build dist src/lingcomp.egg-info
	make build
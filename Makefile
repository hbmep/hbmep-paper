SHELL := /bin/bash
CWD := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

export

python = 3.11

.PHONY: build-base
build-base:
	@python$(python) -m venv .venv

.PHONY: build
build: build-base
	@source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -e . && \
	pip install -e ../hbmep


notebook:
	@source .venv/bin/activate && \
	jupyter notebook

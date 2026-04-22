FROM python:3.10.12-slim-bullseye

ENV HOME=/workspace
ENV PIP_NO_CACHE_DIR=off

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR ${HOME}

RUN set -eux && apt-get update && apt-get install -y --no-install-recommends git curl build-essential procps && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ${HOME}/
RUN pip install --no-cache-dir poetry && poetry install --no-root --only main

COPY . ${HOME}/
ENV PYTHONPATH=${HOME}/python

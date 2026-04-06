# syntax=docker/dockerfile:1


FROM ghcr.io/astral-sh/uv:0.10.4 AS uvbin


FROM nvcr.io/nvidia/pytorch:24.09-py3 AS develop

COPY --from=uvbin /uv /uvx /usr/local/bin/

ARG USERNAME=dcuser
ARG UID=1000
ARG GID=1000

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    DEBIAN_FRONTEND=noninteractive

ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV PYTHONPATH="/workspace:${PYTHONPATH}:"

RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections

RUN --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \

    curl \
    gnupg \
    git \
    git-lfs \
    gh \

    fontconfig \
    git \
    ttf-mscorefonts-installer \
    && git lfs install --system \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN fc-cache -fv

RUN npm i -g @openai/codex

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install torchaudio==2.7.0

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=bind,source=.devcontainer/requirements-dev.txt,target=/tmp/requirements-dev.txt \
    python -m pip install -r /tmp/requirements-dev.txt

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --no-cache-dir ruff

RUN mkdir -p /ms-playwright \
    && python -m playwright install --with-deps chromium \
    && chmod -R a+rX /ms-playwright

RUN addgroup --gid $GID $USERNAME \
    && adduser --disabled-password --gecos "" --shell /bin/bash --uid $UID --gid $GID $USERNAME \
    && chown -R $UID:$GID /ms-playwright

USER $USERNAME
ENV HOME=/home/$USERNAME \
    CODEX_HOME=/home/$USERNAME/.codex \
    PYTHONPATH="${PYTHONPATH}:/workspace/konietse-DAS-CN2S-cb0ee28:/workspace"

ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
WORKDIR /workspace

COPY --chown=$UID:$GID ruff.toml /home/$USERNAME/

CMD ["/bin/bash"]

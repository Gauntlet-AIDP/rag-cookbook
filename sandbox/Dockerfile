FROM python:3.12-slim@sha256:0ced713d9b8336172d981d8f25f2422b6782bdad019fda4cb653c6ee187430de

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.lock /tmp/requirements.lock
# CPU torch comes from the PyTorch CPU index; every artifact is still hash-checked
RUN pip install --require-hashes --no-deps \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r /tmp/requirements.lock

# Bake the embedding model at the pinned revision, safetensors only
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('BAAI/bge-small-en-v1.5', revision='5c38ec7c405ec4b44b94cc5a9bb96e735b38267a', \
local_dir='/models/bge-small', \
allow_patterns=['*.json','*.txt','model.safetensors','1_Pooling/*'])" \
 && ls -R /models/bge-small | head -30 \
 && chmod -R a-w /models

RUN useradd --create-home --uid 10001 app && mkdir /work && chown app /work

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    LANGCHAIN_TRACING_V2=false \
    PYTHONPATH=/shim

USER app
WORKDIR /work
CMD ["bash"]

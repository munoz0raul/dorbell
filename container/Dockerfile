# Dragonwing Doorbell — container image (Stage 2, IQ-8275 + Ubuntu 24.04)
#
# Build from the repo ROOT so doorbell.py is in the build context:
#   docker build -t dragonwing-doorbell -f container/Dockerfile .
#
# The NPU (HTP v75) is reached at runtime through a Container Device Interface
# (CDI) file — see container/cdi/ and container/README.md. The QNN runtime libs
# are NOT installed here: the `qai-appbuilder` wheel bundles its own matching
# QNN libraries + HTP v75 skels, so the image stays driver-free and the CDI only
# forwards the kernel-coupled FastRPC/DMA plumbing from the host.
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

# Match the host OS (ubuntu:24.04) for ABI compatibility with the bind-mounted
# host libraries the CDI forwards (libcdsprpc.so, the DSP skels, etc.).
# libatomic1 is required by qai-appbuilder's bundled QNN native libs (they link
# libatomic.so.1, which is NOT in a minimal ubuntu:24.04) — without it the
# qai_appbuilder import fails and the NPU path silently falls back to CPU.
# libyaml-0-2 is required on **Qualcomm Linux (QLI/Yocto)** hosts: there the
# bind-mounted host libcdsprpc.so additionally links libyaml-0.so.2, absent from
# minimal ubuntu:24.04 — without it the CDI-mounted FastRPC stub fails to load
# and the NPU falls back to CPU. Harmless on Ubuntu hosts (which don't need it).
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-venv \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      libatomic1 \
      libyaml-0-2 \
      v4l-utils \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# App Python dependencies. qai-appbuilder provides the in-process NPU runtime.
COPY container/requirements.txt /tmp/requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY doorbell.py /app/doorbell.py

# Models are mounted at runtime under /root (see run_doorbell_container.sh):
#   /root/qnn_context/*.bin        NPU HTP context binaries
#   /root/.insightface/models/...  CPU ONNX models (buffalo_sc)
# They are board-specific and large, so they are never baked into the image.

ENTRYPOINT ["python3", "/app/doorbell.py"]
CMD ["--backend", "npu"]

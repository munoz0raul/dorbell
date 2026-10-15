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
#
# TWO hard constraints pin the base image and the Python version together:
#
#   1. GLIBC floor. The host libcdsprpc.so forwarded by the CDI links
#      GLIBC_2.38, so the container base MUST provide >= 2.38. ubuntu:24.04 has
#      GLIBC 2.39 (ubuntu:22.04 only has 2.35 → the FastRPC stub fails to load
#      with `loadRemoteSymbols failed err 4000` / `Transport setup 14001` and
#      the NPU silently falls back to CPU). So the base stays ubuntu:24.04.
#
#   2. qai-appbuilder availability. The committed qnn_context/*.bin are generated
#      at QAIRT 2.50, so the host SDK must be a 2.50.x qai-appbuilder (2.48 is
#      too old → `Using newer context binary on old SDK` err 5000; 2.51 skel is
#      rejected by 2.50 DSP firmware → `openSessionForPriority status=0x72`).
#      PyPI NO LONGER ships any 2.50.x wheel (2.50.40 was pulled; only 2.40.0,
#      2.48.40, 2.51.40 remain). The ONLY public 2.50.x build is Qualcomm's
#      GitHub release qualcomm/qai-appbuilder v2.50.0 — and it is a **cp310**
#      wheel (no cp312).
#
# Reconciling the two: ubuntu:24.04's system Python is 3.12 (no public 2.50.x
# cp312 wheel exists), so we add deadsnakes python3.10 and run the app from a
# python3.10 venv into which the public cp310 v2.50.0 wheel installs. The system
# python3.12 is left untouched; GLIBC stays 2.39. Verified on a freshly flashed
# QLI 2.1 IQ-8275 (firmware 2.50.0.260828): real ~300ms HTP load, FastRPC + DMA
# fds held open, no 0x72 / err 4000 / newer-context errors.
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

# System libs. libatomic1: qai-appbuilder's native libs link libatomic.so.1,
# absent from minimal ubuntu:24.04 — without it `import qai_appbuilder` throws
# and the NPU path falls back to CPU. libyaml-0-2: on Qualcomm Linux (QLI/Yocto)
# hosts the bind-mounted host libcdsprpc.so additionally links libyaml-0.so.2,
# absent from minimal ubuntu:24.04 — without it the CDI-mounted FastRPC stub
# fails to load (harmless on Canonical Ubuntu hosts, which don't need it).
# deadsnakes/ppa supplies python3.10 (for the cp310 QNN wheel) alongside the
# stock python3.12, which stays the system default.
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common \
      ca-certificates && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
      python3.10 \
      python3.10-venv \
      python3.10-dev \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      libatomic1 \
      libyaml-0-2 \
      v4l-utils && \
    rm -rf /var/lib/apt/lists/*

# Isolated python3.10 venv for the app. /opt/venv/bin goes first on PATH so the
# container's `python3` is the 3.10 venv interpreter (the QNN runtime lives here).
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# App Python dependencies (pure-pip; no QNN SDK). onnxruntime/numpy are given as
# ranges, not exact pins, because the cp310 wheel set differs from cp312 and
# qai-appbuilder caps numpy<=2.4.4; pip resolves a compatible trio.
COPY container/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# qai-appbuilder: the host-side QNN runtime bundling the HTP v75 skel/stub. It
# must be a 2.50.x build to match the generation of qnn_context/*.bin (see the
# header). Installed from Qualcomm's PUBLIC GitHub release (PyPI dropped 2.50.x),
# cp310 wheel into the 3.10 venv. Override only to run .bin REGENERATED at a
# different SDK — pass a full wheel URL, e.g.:
#   docker build --build-arg QAI_APPBUILDER_WHEEL=\
#     https://github.com/qualcomm/qai-appbuilder/releases/download/v2.51.40/qai_appbuilder-2.51.40-cp310-cp310-manylinux_2_39_aarch64.whl ...
ARG QAI_APPBUILDER_WHEEL=https://github.com/qualcomm/qai-appbuilder/releases/download/v2.50.0/qai_appbuilder-2.50.0-cp310-cp310-linux_aarch64.whl
RUN pip install --no-cache-dir "$QAI_APPBUILDER_WHEEL"

WORKDIR /app
COPY doorbell.py /app/doorbell.py

# Models are mounted at runtime under $HOME (see the run wrappers):
#   ~/qnn_context/*.bin        NPU HTP context binaries
#   ~/.insightface/models/...  CPU ONNX models (buffalo_sc)
# They are board-specific and large, so they are never baked into the image.

ENTRYPOINT ["python3", "/app/doorbell.py"]
CMD ["--backend", "npu"]

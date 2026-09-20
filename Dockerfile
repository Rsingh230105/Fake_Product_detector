# FROM python:3.10.13-slim

# # Python ko .pyc files na banane do, aur logs turant screen pe dikhein (buffer na ho)
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# # System-level dependencies (pip se nahi aate, apt se install hote hain)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     tesseract-ocr \
#     libzbar0 \
#     libmagic1 \
#     libgl1 \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Pehle sirf requirement files copy karo (layer caching ke liye — agla topic mein detail)
# COPY requirements.txt ./
# RUN pip install --upgrade pip && \
#     pip install -r requirements.txt && \
#     pip install python-magic==0.4.27
    
# # Ab poora project copy karo (.dockerignore jo exclude karega, wo skip hoga)
# COPY . .
# RUN chmod +x entrypoint.sh

# # Security: root user se mat chalao — non-root user banao
# RUN useradd -m appuser && \
#     mkdir -p /app/models && \
#     chown -R appuser:appuser /app
# USER appuser

# EXPOSE 8000

# ENTRYPOINT ["./entrypoint.sh"]

###----------------1.46 gb current image size--------------------------###################--------------------------------

## Use Multi stage 

# # ============================================================================
# # STAGE 1: "builder" — used only to install dependencies
# # This entire stage is discarded and does NOT end up in the final image
# # ============================================================================
# FROM python:3.10.13-slim AS builder

# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# # Compiler tools — some Python packages (e.g. Levenshtein, pyzbar) need to be
# # compiled from source if no pre-built wheel is available for this platform.
# # This is only needed in THIS stage — it will never be part of the final image.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Copy only the requirements file first (enables Docker layer caching)
# COPY requirements.txt ./

# # --user is important: packages get installed into /root/.local
# # (instead of system-wide /usr/local), so in Stage 2 we only need to
# # copy this one folder — not the whole Python environment.
# # --no-cache-dir: prevents pip from keeping its download cache on disk (saves space)
# RUN pip install --upgrade pip && \
#     pip install --user --no-cache-dir -r requirements.txt && \
#     pip install --user --no-cache-dir python-magic==0.4.27


# # ============================================================================
# # STAGE 2: Runtime — this IS the final production image (small, secure)
# # Fresh start — none of Stage 1's build-essential, pip cache, etc. carry over
# # ============================================================================
# FROM python:3.10.13-slim

# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1
# # Tell Python where to find the --user-installed packages/binaries
# ENV PATH=/home/appuser/.local/bin:$PATH

# # These are RUNTIME libraries only (.so files) — no compiling needed here,
# # just needed to run the program. That's why build-essential is NOT here.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     tesseract-ocr \
#     libzbar0 \
#     libmagic1 \
#     libgl1 \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Create the non-root user upfront (wasn't needed in Stage 1 since no app code was running there)
# RUN useradd -m appuser

# # This is the core of multi-stage builds: copy ONLY the installed packages
# # from the builder stage — build-essential, apt cache, pip cache, and any
# # intermediate files are left behind. --chown sets ownership in the same step.
# COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# # Now copy the full project code (anything in .dockerignore is skipped)
# COPY --chown=appuser:appuser . .
# RUN chmod +x entrypoint.sh

# # Pre-create the model download folder and confirm ownership
# RUN mkdir -p /app/models && chown -R appuser:appuser /app

# # Run as non-root from here on (security best practice)
# USER appuser

# EXPOSE 8000

# ENTRYPOINT ["./entrypoint.sh"]


###############################3
# ============================================================================
# STAGE 1: Builder
# ============================================================================

FROM python:3.10-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Upgrade Python packaging tools
RUN python -m pip install --upgrade \
        pip \
        setuptools \
        wheel && \
    python -m pip install \
        --user \
        --no-cache-dir \
        -r requirements.txt && \
    python -m pip install \
        --user \
        --no-cache-dir \
        python-magic==0.4.27


# ============================================================================
# STAGE 2: Runtime
# ============================================================================

FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH=/home/appuser/.local/bin:$PATH

# Runtime libraries only
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libzbar0 \
        libmagic1 \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Non-root user
RUN useradd -m appuser

# Copy Python packages
COPY --from=builder \
    --chown=appuser:appuser \
    /root/.local \
    /home/appuser/.local

# Copy application
COPY --chown=appuser:appuser . .

RUN chmod +x entrypoint.sh

RUN mkdir -p /app/models && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
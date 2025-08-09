FROM nvidia/cuda:12.8.61-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git python3 python3-pip wget curl unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy start script
COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

# Expose ComfyUI port
EXPOSE 8188

CMD ["/workspace/start.sh"]

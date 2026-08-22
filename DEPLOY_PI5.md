# Raspberry Pi 5 Deployment Guide

Deploy CortexEdge on a Raspberry Pi 5 (8GB RAM) for local LLM inference
with adaptive routing, RAG, and offline speech.

---

## Hardware Requirements

| Component | Specification | Purpose |
|---|---|---|
| **Raspberry Pi 5** | 8GB RAM | Primary compute |
| **MicroSD Card** | 64GB+ (A2/U3) | OS + models (or USB 3 SSD for faster loading) |
| **Power Supply** | 27W USB-C (5V/5A) | Required for full performance |
| **Active Cooling** | Heatsink + fan | Prevents thermal throttling |
| **USB Mic** | Any USB microphone | STT (optional, Phase 9) |
| **USB Speaker** | Any 3.5mm/USB speaker | TTS output (optional) |
| **INA219 Sensor** | I2C breakout board | Energy measurement (optional) |

### INA219 Wiring (optional)
| INA219 Pin | Pi Pin |
|---|---|
| VIN | 5V |
| GND | GND |
| SDA | GPIO 2 (SDA1) |
| SCL | GPIO 3 (SCL1) |

---

## Step 1: Flash 64-bit Raspberry Pi OS

1. Download **Raspberry Pi OS (64-bit)** from [raspberrypi.com](https://www.raspberrypi.com/software/)
2. Flash to SD card using Raspberry Pi Imager
3. Enable before first boot:
   - SSH (Settings gear icon)
   - WiFi credentials
   - Username/password
   - Locale/timezone

---

## Step 2: Initial Setup

```bash
# SSH into the Pi
ssh pi@<pi-ip-address>

# Update system
sudo apt update && sudo apt full-upgrade -y

# Install system dependencies
sudo apt install -y \
  git cmake build-essential \
  python3-pip python3-venv \
  ffmpeg libopenblas-dev \
  sqlite3 libraspberrypi-bin

# Reboot after kernel updates
sudo reboot
```

---

## Step 3: Install Ollama (ARM64)

```bash
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version

# Enable auto-start on boot
sudo systemctl enable ollama
```

---

## Step 4: Pull Models

Ollama loads models on demand and unloads after idle. Total disk usage: ~5.5GB.

```bash
# Small tier — fast, ~0.7s response (~400MB)
ollama pull qwen2.5:0.5b

# Medium tier — quality, ~3s response (~1.6GB)
ollama pull gemma2:2b

# MoE tier — efficient, ~1.5s response (~1GB)
ollama pull qwen2.5:1.5b

# Large tier — highest quality, slow on CPU (~2.5GB)
ollama pull qwen3:4b
```

### Memory budget (8GB)
| Component | RAM |
|---|---|
| OS + services | ~1.5GB |
| Small model | ~0.5GB |
| Medium model | ~2.0GB |
| Large model | ~3.0GB |
| **Total (all loaded)** | **~7GB** |

> **Tip:** Ollama keeps only one model loaded at a time by default. The
> router adapts under memory pressure — if you want multiple models
> loaded simultaneously, set `OLLAMA_NUM_PARALLEL=1` and be aware of
> swap usage.

---

## Step 5: Clone & Configure

```bash
cd ~
git clone https://github.com/<your-username>/edge-cortex.git
cd edge-cortex

# Create Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Create `.env`

```bash
cp .env.example .env
```

Edit `.env` for Pi 5:

```env
# Inference backend
BACKEND=ollama
OLLAMA_API_URL=http://localhost:11434

# Model names (must match pulled models)
MODEL_SMALL=qwen2.5:0.5b
MODEL_MEDIUM=gemma2:2b
MODEL_MOE=qwen2.5:1.5b
MODEL_LARGE=qwen3:4b

# Routing defaults
DEFAULT_ROUTER=rule
DEFAULT_POLICY=public
QUALITY_TARGET=0.6

# Resource limits
MAX_CPU_TEMPERATURE_C=75.0
MINIMUM_FREE_RAM_MB=1500
REQUEST_TIMEOUT_S=90

# RAG
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_GENERATION_ROUTE=medium
RAG_CHUNK_SIZE=450
RAG_CHUNK_OVERLAP=70

# Cache
SEMANTIC_CACHE_THRESHOLD=0.95
```

---

## Step 6: Run CortexEdge

### API Server
```bash
make run
# or: uvicorn app.api.server:app --host 0.0.0.0 --port 8000
```

### Streamlit UI (in another terminal)
```bash
make ui
# Access at http://<pi-ip>:8501
```

### Verify
```bash
curl http://localhost:8000/health
```

Expected output:
```json
{
  "status": "ok",
  "system": {
    "free_ram_mb": 6000.0,
    "cpu_percent": 10.0,
    "cpu_temperature_c": 50.0,
    "network_available": true
  },
  "routes": {
    "small": "qwen2.5:0.5b",
    "medium": "gemma2:2b",
    "moe": "qwen2.5:1.5b",
    "large": "qwen3:4b"
  },
  "enabled_routes": ["large", "medium", "moe", "small"]
}
```

---

## Step 7: Test

### Simple query
```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"text":"What is 2+2?","router":"rule","policy":"public"}'
```

### Streaming
```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H 'Content-Type: application/json' \
  -d '{"text":"Explain what a GPU does.","model":"small","max_tokens":256}'
```

### With RAG
```bash
# First, ingest documents
mkdir -p datasets/documents
cp your-documents/*.pdf datasets/documents/
make ingest

# Query
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"text":"What does the document say about X?","use_documents":true}'
```

---

## Optional: INA219 Power Logging

For measuring energy consumption per query:

```bash
# Install Adafruit library
pip install adafruit-circuitpython-ina219

# Wire INA219 to I2C (see hardware table above)
# Verify I2C connection
sudo i2cdetect -y 1

# Run power logger
python3 scripts/log_power.py
```

Output: `experiments/results/power.csv` with voltage, current, power, and
cumulative energy in joules.

---

## Optional: Whisper STT + Piper TTS

For voice interaction (requires USB mic + speaker):

```bash
# Build whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make
cd ..

# Download whisper model
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
mv ggml-base.en.bin models/stt/

# Install Piper TTS
pip install piper-tts

# Download Piper voice
wget https://github.com/rhasspy/piper/releases/download/v2023.11.1-2/en_US-lessac-medium.onnx
mv en_US-lessac-medium.onnx models/tts/
```

---

## Troubleshooting

### Swap usage / OOM
```bash
# Check swap
free -h

# Increase swap to 2GB (if needed)
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Thermal throttling
```bash
# Check temperature
vcgencmd measure_temp

# Set active cooling threshold (in /boot/config.txt)
# gpu_temp=75,gpu_fan=255  # Fan at 100% above 75°C
sudo reboot
```

### Model loading slow
```bash
# Use USB 3 SSD instead of microSD for model storage
# Move Ollama models to SSD
sudo systemctl stop ollama
sudo mv /usr/share/ollama/.ollama /mnt/ssd/ollama
sudo ln -s /mnt/ssd/ollama /usr/share/ollama/.ollama
sudo systemctl start ollama
```

### API unreachable from other devices
```bash
# Ensure uvicorn binds to 0.0.0.0, not 127.0.0.1
# In Makefile or .env:
#   --host 0.0.0.0

# Check firewall
sudo ufw allow 8000/tcp
sudo ufw allow 8501/tcp
```

### Embeddings torch errors (no GPU)
```bash
# PyPI torch ships CUDA binaries that fail on CPU-only machines
pip install torch --index-url https://download.pytorch.org/whl/cpu
# Then re-run ingest
make ingest
```

---

## Recommended Model Tiers for Pi 5 (8GB)

| Tier | Model | Size | Latency | Quality |
|---|---|---|---|---|
| small | `qwen2.5:0.5b` | 400MB | ~0.7s | Basic |
| medium | `gemma2:2b` | 1.6GB | ~3s | Good |
| moe | `qwen2.5:1.5b` | 1GB | ~1.5s | Good |
| large | `qwen3:4b` | 2.5GB | ~8s | Best |

**Recommended setup:** Keep small + medium always available. The router
auto-selects based on query complexity, device state, and policy.

---

## Production Tips

1. **Run as a systemd service** for auto-restart:
   ```bash
   sudo tee /etc/systemd/system/cortexedge.service << 'EOF'
   [Unit]
   Description=CortexEdge API
   After=network.target ollama.service

   [Service]
   Type=simple
   User=pi
   WorkingDirectory=/home/pi/edge-cortex
   ExecStart=/home/pi/edge-cortex/.venv/bin/uvicorn app.api.server:app --host 0.0.0.0 --port 8000
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   EOF

   sudo systemctl daemon-reload
   sudo systemctl enable cortexedge
   sudo systemctl start cortexedge
   ```

2. **Monitor logs:**
   ```bash
   journalctl -u cortexedge -f
   ```

3. **Auto-update on boot:**
   ```bash
   # Add to /etc/rc.local before exit 0
   cd /home/pi/edge-cortex && git pull && source .venv/bin/activate && pip install -r requirements.txt
   ```

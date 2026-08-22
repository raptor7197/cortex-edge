"""System state probes: memory, CPU, temperature, network (PLAN 15.4).

Device-state features feed the router and the experiment logger.
"""

import socket
from pathlib import Path

import psutil

from app.config import settings
from app.schemas import SystemState

THERMAL_PATH = Path("/sys/class/thermal/thermal_zone0/temp")


def read_temperature() -> float | None:
    try:
        return float(THERMAL_PATH.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def network_available(host: str = "1.1.1.1", port: int = 53) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def loaded_models() -> list[str]:
    """Models currently loaded in the inference backend (Ollama)."""
    if settings.backend != "ollama":
        return []
    try:
        import requests

        r = requests.get(
            f"{settings.ollama_api_url}/api/ps", timeout=2
        )
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def read_system_state() -> SystemState:
    vm = psutil.virtual_memory()
    return SystemState(
        free_ram_mb=vm.available / (1024 * 1024),
        cpu_percent=psutil.cpu_percent(interval=0.1),
        cpu_temperature_c=read_temperature(),
        network_available=network_available(),
        loaded_models=loaded_models(),
    )

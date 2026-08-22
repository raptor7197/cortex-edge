"""Model manager (PLAN 16.3): memory-safe load/unload rules.

With Ollama the backend keeps models resident automatically; the
manager still tracks residency and unloads models under memory
pressure. With llama.cpp (Pi path) each tier runs as its own
`llama-server` subprocess; the manager starts/stops those processes
based on free RAM.

Rules:
  - never load a model whose estimated memory leaves less than
    `minimum_free_ram_mb` available
  - under pressure, unload the largest resident model first
"""

import subprocess
import time

import psutil

from app.config import settings
from app.inference import llm_client


class ModelManager:
    def __init__(self):
        self._subprocesses: dict[str, subprocess.Popen] = {}
        self._manual: set[str] = set()

    # --- Ollama backend ----------------------------------------------------

    def resident_models(self) -> list[str]:
        """Models currently loaded in Ollama (by route name, best effort)."""
        try:
            from app.monitoring.system_state import loaded_models

            names = set(loaded_models())
            return sorted(
                route
                for route, model in llm_client.LOCAL_MODELS.items()
                if model in names or any(n.startswith(model) for n in names)
            )
        except Exception:
            return []

    def unload(self, route: str):
        """Unload a model from Ollama (keep_alive=0)."""
        if settings.backend != "ollama":
            return
        import requests

        requests.post(
            f"{settings.ollama_api_url}/api/chat",
            json={
                "model": llm_client.local_model_name(route),
                "messages": [{"role": "user", "content": "x"}],
                "options": {"keep_alive": 0},
            },
            timeout=5,
        )
        self._manual.discard(route)

    def prune(self):
        """Unload the largest resident model if free RAM is too low."""
        if settings.backend != "ollama":
            return
        import requests

        r = requests.get(f"{settings.ollama_api_url}/api/ps", timeout=3)
        r.raise_for_status()
        models = sorted(
            r.json().get("models", []),
            key=lambda m: m.get("size_vram", 0),
            reverse=True,
        )
        free_mb = psutil.virtual_memory().available / (1024 * 1024)
        if free_mb >= settings.minimum_free_ram_mb:
            return
        for m in models:
            if free_mb >= settings.minimum_free_ram_mb:
                break
            try:
                requests.post(
                    f"{settings.ollama_api_url}/api/chat",
                    json={
                        "model": m["name"],
                        "messages": [{"role": "user", "content": "x"}],
                        "options": {"keep_alive": 0},
                    },
                    timeout=5,
                )
                free_mb += m.get("size_vram", 0) / (1024 * 1024)
            except Exception:
                pass

    def fits(self, route: str) -> bool:
        """True if loading this model keeps enough free RAM (estimate)."""
        from app.router.cost import route_cost_for

        cost = route_cost_for(route)
        if cost is None:
            return True
        free_mb = psutil.virtual_memory().available / (1024 * 1024)
        return free_mb - cost.memory_mb >= settings.minimum_free_ram_mb

    # --- llama.cpp backend (Raspberry Pi path) ------------------------------

    def ensure_server(self, route: str):
        """Start a llama-server subprocess for the route if not running."""
        if settings.backend != "ollama" and route not in self._subprocesses:
            raise RuntimeError(
                "llama.cpp subprocess management is not configured; "
                "start llama-server manually (see PLAN section 15.1) "
                "or use the Ollama backend"
            )

    def shutdown(self):
        for proc in self._subprocesses.values():
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                pass
        self._subprocesses.clear()

    def wait_until_loaded(self, route: str, timeout_s: int = 60) -> bool:
        """Block until the route's model is resident (or timeout)."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if route in self.resident_models():
                return True
            time.sleep(1)
        return route in self.resident_models()

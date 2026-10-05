.PHONY: setup run ui test benchmark ingest dataset train power models api-restart smoke \
        bench-latency bench-e2e pooled-serve pooled-bridge verify-pooled

PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

setup:
	$(PY) -m pip install -r requirements.txt

models:
	bash scripts/start_ollama_local.sh
	bash scripts/import_local_models.sh

run:
	uvicorn app.api.server:app --host 127.0.0.1 --port 8000

api-restart:
	bash scripts/restart_api.sh

ui:
	streamlit run app/ui.py

test:
	$(PY) -m pytest tests/ -q

smoke:
	bash scripts/smoke_test.sh

benchmark:
	$(PY) scripts/benchmark_routes.py

bench-latency:
	$(PY) scripts/bench_latency.py

bench-e2e:
	$(PY) scripts/bench_end_to_end.py

# pooled (peer-to-peer browser inference): POOLED_ROOM_LINK=... make pooled-serve
pooled-serve:
	bash scripts/pooled_serve.sh "$(POOLED_ROOM_LINK)"

pooled-bridge:
	$(PY) scripts/pooled_mock_bridge.py --port 8080 --upstream-model $(or $(UPSTREAM),qwen2.5:1.5b)

verify-pooled:
	bash scripts/verify_pooled_route.sh

dataset:
	$(PY) scripts/build_dataset.py

train:
	$(PY) scripts/train_router.py

ingest:
	$(PY) scripts/ingest_docs.py

power:
	$(PY) scripts/log_power.py

.PHONY: setup run ui test benchmark ingest dataset train power

setup:
	python3 -m pip install -r requirements.txt

run:
	uvicorn app.api.server:app --host 127.0.0.1 --port 8000

ui:
	streamlit run app/ui.py

test:
	python3 -m pytest tests/ -q

benchmark:
	python3 scripts/benchmark_routes.py

dataset:
	python3 scripts/build_dataset.py

train:
	python3 scripts/train_router.py

ingest:
	python3 scripts/ingest_docs.py

power:
	python3 scripts/log_power.py
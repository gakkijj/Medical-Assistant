.PHONY: benchmark test-offline run docker

PYTHON ?= python3

benchmark:
	cd medix-agent-swarm && $(PYTHON) evaluation/run_benchmark.py --check

test-offline:
	cd medix-agent-swarm && $(PYTHON) -m unittest discover -s tests -p 'test_offline_*.py' -v

run:
	cd medix-agent-swarm && uvicorn api.app:app --host 127.0.0.1 --port 8000

docker:
	docker compose up --build

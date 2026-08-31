.PHONY: evaluate test-ml validate-real evaluate-real site-build api

evaluate:
	python3 -m ml.scripts.evaluate_seeded --seed 5050 --threshold 0.68

test-ml:
	python3 -m unittest discover -s ml/tests -v

validate-real:
	python3 -m ml.scripts.validate_real_dataset data/real/manifest.jsonl --check-files

evaluate-real:
	python3 -m ml.scripts.evaluate_real_predictions artifacts/real-predictions.jsonl --json-output artifacts/real-report.json --markdown-output artifacts/real-report.md

site-build:
	npm run build

api:
	python3 -m uvicorn service.app:app --host 127.0.0.1 --port 8000

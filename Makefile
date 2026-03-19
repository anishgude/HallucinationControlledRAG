setup:
	python -m pip install -r requirements.txt

experiment:
	python make_dataset.py
	python run_experiment.py
	python evaluate.py

report:
	python evaluate.py

all:
	$(MAKE) setup
	$(MAKE) clean
	$(MAKE) train_classifier
	$(MAKE) test_classifier
	$(MAKE) train_stories
	$(MAKE) test_stories

clean:
	rm -rf models/

retrain:
	$(MAKE) clean
	$(MAKE) train_classifier
	$(MAKE) train_stories

setup:
	pip install -r requirements.txt

train_classifier:
	python -m rasa_nlu.train -c nlu_config.yml --data nlu.md -o models --fixed_model_name nlu --project current --verbose

test_classifier:
	python test_classifier.py

train_stories:
	python -m rasa_core.train -d domain.yml -s stories.md -o models/dialogue

test_stories:
	python test_stories.py

	
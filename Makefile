all:
	$(MAKE) run

setup: 
	source bin/activate
	pip install --user --upgrade tensorflow
	virtualenv --system-site-packages -p python3 $PWD
	export TF_CPP_MIN_LOG_LEVEL=2

board:
	tensorboard --logdir=tmp/

run:
	# python 1_getting_started.py
	# python 2_custom_model.py
	# python 3_mnist_softmax.py
	# python 4_multi_CNN.py
	python 5_iris_classifier.py

install_env: # for osx
	sudo easy_install pip
	pip install --user --upgrade virtualenv
	alias virtualenv='~/Library/Python/2.7/bin/virtualenv' 
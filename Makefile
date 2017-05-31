all:
	$(MAKE) run

setup: 
	source bin/activate
	export TF_CPP_MIN_LOG_LEVEL=2

board:
	tensorboard --logdir=tmp/
	# tensorboard --logdir=tmp/iris_model/

run:
	# python 1_getting_started.py
	# python 2_custom_model.py
	# python 3_mnist_softmax.py
	# python 4_multi_CNN.py
	# python 5_iris_classifier.py
	# python 6_word2vec_basic.py
	# python 7_lstm_fixed_sequence_length.py
	python 8_lstm_variable_sequence_length.py

#magenta
magenta_install:
	pip install magenta

magenta_cr_dataset:
	convert_dir_to_note_sequences \
	  --input_dir=tmp/TouhouMidiMusic/ \
	  --output_file=tmp/notesequences.tfrecord \
	  --recursive


magenta_melo_gen:
	melody_rnn_generate \
		--config=lookback_rnn \
		--bundle_file=tmp/magenta-models/lookback_rnn.mag \
		--output_dir=tmp/magenta-data/lookback_rnn/generated \
		--num_outputs=10 \
		--num_steps=128 \
		--primer_melody="[60, -2, 60, -2, 67, -2, 67, -2]"

magenta_poly_cr_data:
	polyphony_rnn_create_dataset \
		--input=tmp/notesequences.tfrecord \
		--output_dir=tmp/polyphony_rnn/sequence_examples \
		--eval_ratio=0.10

magenta_poly_gen:
	polyphony_rnn_generate \
		--output_dir=tmp/polyphony_rnn/generated \
		--num_outputs=10 \
		--num_steps=128 \
		--primer_pitches="[67,64,60]" \
		--condition_on_primer=true \
		--inject_primer_during_generation=false \
		--hparams="{'batch_size':64,'rnn_layer_sizes':[64,64]}" \
		--run_dir=tmp/polyphony_rnn/logdir/run1
		# --bundle_file=tmp/magenta-models/polyphony_rnn.mag

magenta_poly_train:
	polyphony_rnn_train \
		--run_dir=tmp/polyphony_rnn/logdir/run1 \
		--sequence_example_file=tmp/polyphony_rnn/sequence_examples/training_poly_tracks.tfrecord \
		--hparams="{'batch_size':64,'rnn_layer_sizes':[64,64]}" \
		--num_training_steps=1000

magenta_poly_train_board:
	tensorboard --logdir=tmp/polyphony_rnn/logdir



# for osx
install_env: 
	sudo easy_install pip
	virtualenv --system-site-packages -p python3 $PWD
	alias virtualenv='~/Library/Python/2.7/bin/virtualenv'
	$(MAKE) setup
	pip install --user --upgrade virtualenv
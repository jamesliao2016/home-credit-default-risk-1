data/prev.preprocessed.feather: data/previous_application.feather preprocess_prev.py
	python preprocess_prev.py
data/prev.last.feather: data/prev.preprocessed.feather create_prev_last.py
	python create_prev_last.py

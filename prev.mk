.PHONY: prev
prev: data/prev.agg.feather data/prev.last.feather

data/prev.preprocessed.feather: data/previous_application.feather preprocess_prev.py
	python preprocess_prev.py
data/prev.last.feather: data/prev.preprocessed.feather create_prev_last.py
	python create_prev_last.py
data/prev.agg.feather: data/prev.preprocessed.feather aggregate_prev.py
	python aggregate_prev.py
data/prev.refused.feather: data/prev.preprocessed.feather ./create_prev_refused.py
	python create_prev_refused.py
data/prev.approved.feather: data/prev.preprocessed.feather ./create_prev_approved.py
	python create_prev_approved.py
data/prev.grp.feather: data/prev.preprocessed.feather ./create_prev_grp.py
	python create_prev_grp.py

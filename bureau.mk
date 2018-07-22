data/bureau_balance.preprocessed.feather: data/bureau_balance.feather preprocess_bb.py
	python preprocess_bb.py

data/bureau_balance.agg.feather: data/bureau_balance.preprocessed.feather aggregate_bb.py
	python aggregate_bb.py

data/bureau.preprocessed.feather: data/bureau_balance.agg.feather preprocess_bureau.py
	python preprocess_bureau.py

data/bureau.agg.feather: data/bureau.preprocessed.feather aggregate_bureau.py
	python aggregate_bureau.py


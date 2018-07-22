data/bureau_balance.preprocessed.feather: data/bureau_balance.feather preprocess_bb.py
	python preprocess_bb.py

data/bb.edge.feather: data/bureau_balance.preprocessed.feather create_edge_bb.py
	python create_edge_bb.py

data/bb.agg.feather: data/bureau_balance.preprocessed.feather data/bb.edge.feather aggregate_bb.py
	python aggregate_bb.py

data/bureau.preprocessed.feather: data/bureau_balance.agg.feather preprocess_bureau.py
	python preprocess_bureau.py

data/bureau.agg.feather: data/bureau.preprocessed.feather aggregate_bureau.py
	python aggregate_bureau.py

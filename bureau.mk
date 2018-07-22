data/bb.preprocessed.feather: data/bureau_balance.feather preprocess_bb.py
	python preprocess_bb.py

data/bb.edge.feather: data/bb.preprocessed.feather create_edge_bb.py
	python create_edge_bb.py

data/bb.agg.feather: data/bb.preprocessed.feather data/bb.edge.feather aggregate_bb.py
	python aggregate_bb.py

data/bureau.preprocessed.feather: data/bb.agg.feather data/bureau.feather preprocess_bureau.py
	python preprocess_bureau.py

data/bureau.agg.num.feather: data/bureau.preprocessed.feather aggregate_bureau.py
	python aggregate_bureau.py

data/bureau.agg.cat.feather: data/bureau.preprocessed.feather aggregate_bureau_category.py
	python aggregate_bureau_category.py


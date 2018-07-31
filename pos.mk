# pos
data/pos.preprocessed.feather: data/POS_CASH_balance.feather preprocess_pos.py
	python preprocess_pos.py

data/pos.agg.feather: data/pos.preprocessed.feather aggregate_pos.py
	python aggregate_pos.py

data/pos.edge.feather: data/pos.preprocessed.feather create_pos_edge.py
	python create_pos_edge.py

data/pos.edge.agg.num.feather: data/pos.edge.feather aggregate_pos_edge_num.py
	python aggregate_pos_edge_num.py

data/pos.edge.agg.cat.feather: data/pos.edge.feather aggregate_pos_edge_cat.py
	python aggregate_pos_edge_cat.py

data/pos.diff.feather: data/pos.preprocessed.feather create_pos_diff.py
	python create_pos_diff.py

data/pos.tail.feather: data/pos.preprocessed.feather create_pos_tail.py
	python create_pos_tail.py

data/pos.last.feather: data/pos.preprocessed.feather create_pos_last.py
	python create_pos_last.py

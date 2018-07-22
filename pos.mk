# pos
data/pos.preprocessed.feather: data/POS_CASH_balance.feather preprocess_pos.py
	python preprocess_pos.py

data/POS_CASH_balance.agg.curr.feather: data/POS_CASH_balance.preprocessed.feather aggregate_pos_curr.py
	python aggregate_pos_curr.py

data/POS_CASH_balance.agg.prev.feather: data/POS_CASH_balance.preprocessed.feather aggregate_pos_prev.py
	python aggregate_pos_prev.py

data/POS_CASH_balance.agg.prev.last.feather: data/POS_CASH_balance.preprocessed.feather aggregate_pos_prev_last.py
	python aggregate_pos_prev_last.py

data/pos.edge.feather: data/pos.preprocessed.feather create_pos_edge.py
	python create_pos_edge.py

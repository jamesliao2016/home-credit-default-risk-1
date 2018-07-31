data/credit.preprocessed.feather: data/credit_card_balance.feather preprocess_credit.py
	python preprocess_credit.py

data/credit.agg.feather: data/credit.preprocessed.feather aggregate_credit.py
	python aggregate_credit.py

data/credit.last.feather: data/credit.preprocessed.feather create_credit_last.py
	python create_credit_last.py

data/credit.prev.last.feather: data/credit.preprocessed.feather create_credit_prev_last.py
	python create_credit_prev_last.py

data/credit.diff.feather: data/credit.preprocessed.feather create_credit_diff.py
	python create_credit_diff.py

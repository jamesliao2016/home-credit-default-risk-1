data/credit.preprocessed.feather: data/credit_card_balance.feather preprocess_credit.py
	python preprocess_credit.py

data/credit.agg.feather: data/credit.preprocessed.feather aggregate_credit_curr.py
	python aggregate_credit_curr.py

data/credit.diff.feather: data/credit.preprocessed.feather create_credit_diff.py
	create_credit_diff.py

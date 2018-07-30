# application
SRC := data/application_train.feather data/application_test.feather
data/application_train.preprocessed.feather data/application_test.preprocessed.feather: $(SRC) preprocess_app.py
	python preprocess_application.py

data/app.agg.feather: data/application_train.preprocessed.feather data/application_test.preprocessed.feather aggregate_app.py
	python aggregate_app.py

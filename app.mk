# application
SRC := data/application_train.feather data/application_test.feather
data/application_train.preprocessed.feather data/application_test.preprocessed.feather: $(SRC) preprocess_application.py
	python preprocess_application.py

data/app.agg.feather: data/application_train.preprocessed.feather data/application_test.preprocessed.feather aggregate_app.py
	python aggregate_app.py

data/app.grp.diff.feather: data/application_train.preprocessed.feather data/application_test.preprocessed.feather ./create_app_grp_diff.py
	python ./create_app_grp_diff.py

data/app.enc.feather: data/application_train.preprocessed.feather data/application_test.preprocessed.feather ./create_app_encode.py
	python ./create_app_encode.py

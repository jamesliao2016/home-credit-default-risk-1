default:
	echo "do nothing"

SRCS := $(shell ls data | grep csv.zip)
SRCS := $(addprefix data/,$(SRCS))
DSTS := $(SRCS:.csv.zip=.feather)
APP_PREP_DSTS := data/application_test.preprocessed.feather data/application_train.preprocessed.feather

.PHONY: all
all: $(APP_PREP_DSTS)

$(DSTS): %.feather: %.csv.zip
	python convert.py --src $< --dst $@

$(APP_PREP_DSTS): $(DSTS)
	python preprocess_application.py

data/bureau_balance.preprocessed.feather: data/bureau_balance.feather preprocess_bb.py
	python preprocess_bb.py

data/bureau_balance.agg.feather: data/bureau_balance.preprocessed.feather aggregate_bb.py
	python aggregate_bb.py

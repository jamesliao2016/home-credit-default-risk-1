default:
	echo "do nothing"

include ./feather.mk
include ./bureau.mk
include ./pos.mk
include ./inst.mk

APP_PREP_DSTS := data/application_test.preprocessed.feather data/application_train.preprocessed.feather

.PHONY: all
all: $(APP_PREP_DSTS)

# application
# split
SPLITS := $(shell seq 0 10)
SPLITS := $(addsuffix .feather,$(SPLITS))
ORIG_SPLITS := $(addprefix data/application_train.split.,$(SPLITS))
APP := data/application_train.feather data/application_test.feather
PREP_SPLITS := $(addprefix data/application_train.preprocessed.split.,$(SPLITS))

.PHONY: split
split: $(PREP_SPLITS)

$(PREP_SPLITS) $(ORIG_SPLITS): $(APP) $(APP_PREP_DSTS) split_train.py
	python split_train.py

# folds
FOLDS := $(shell seq 0 4)
FOLDS := $(addsuffix .feather,$(FOLDS))
FOLDS := $(addprefix data/fold.,$(FOLDS))

.PHONY: fold
fold: $(FOLDS)
$(FOLDS): $(APP_PREP_DSTS) split_fold.py
	python split_fold.py

# application
$(APP_PREP_DSTS): $(DSTS) preprocess_application.py
	python preprocess_application.py

data/application.agg.feather: data/application_train.preprocessed.feather data/application_test.preprocessed.feather aggregate_app.py
	python aggregate_app.py


# credit
data/credit_card_balance.preprocessed.feather: data/credit_card_balance.feather preprocess_credit.py
	python preprocess_credit.py

data/credit_card_balance.agg.prev.feather: data/credit_card_balance.preprocessed.feather aggregate_credit_prev.py
	python aggregate_credit_prev.py

data/credit_card_balance.agg.prev.last.feather: data/credit_card_balance.preprocessed.feather aggregate_credit_prev_last.py
	python aggregate_credit_prev_last.py

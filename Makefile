default:
	echo "do nothing"

SRCS := $(shell ls data | grep csv.zip)
SRCS := $(addprefix data/,$(SRCS))
DSTS := $(SRCS:.csv.zip=.feather)
APP_PREP_DSTS := data/application_test.preprocessed.feather data/application_train.preprocessed.feather

.PHONY: all
all: $(APP_PREP_DSTS)

# application
$(DSTS): %.feather: %.csv.zip
	python convert.py --src $< --dst $@

SPLITS := $(shell seq 0 10)
SPLITS := $(addsuffix .feather,$(SPLITS))
ORIG_SPLITS := $(addprefix data/application_train.split.,$(SPLITS))
APP := data/application_train.feather data/application_test.feather
PREP_SPLITS := $(addprefix data/application_train.preprocessed.split.,$(SPLITS))
$(PREP_SPLITS) $(ORIG_SPLITS): $(APP) $(APP_PREP_DSTS) split_train.py
	python split_train.py

$(APP_PREP_DSTS): $(DSTS) preprocess_application.py
	python preprocess_application.py

# bureau
data/bureau_balance.preprocessed.feather: data/bureau_balance.feather preprocess_bb.py
	python preprocess_bb.py

data/bureau_balance.agg.feather: data/bureau_balance.preprocessed.feather aggregate_bb.py
	python aggregate_bb.py

data/bureau.preprocessed.feather: data/bureau_balance.agg.feather preprocess_bureau.py
	python preprocess_bureau.py

data/bureau.agg.feather: data/bureau.preprocessed.feather aggregate_bureau.py
	python aggregate_bureau.py

# credit
data/credit_card_balance.preprocessed.feather: data/credit_card_balance.feather preprocess_credit.py
	python preprocess_credit.py

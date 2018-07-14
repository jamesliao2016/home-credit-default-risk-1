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

# application
$(APP_PREP_DSTS): $(DSTS) preprocess_application.py
	python preprocess_application.py

data/application.agg.feather: data/application_train.preprocessed.feather data/application_test.preprocessed.feather aggregate_app.py
	python aggregate_app.py

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

data/credit_card_balance.agg.prev.feather: data/credit_card_balance.preprocessed.feather aggregate_credit_prev.py
	python aggregate_credit_prev.py

data/credit_card_balance.agg.prev.last.feather: data/credit_card_balance.preprocessed.feather aggregate_credit_prev_last.py
	python aggregate_credit_prev_last.py

# pos
data/POS_CASH_balance.preprocessed.feather: data/POS_CASH_balance.feather preprocess_pos.py
	python preprocess_pos.py

data/POS_CASH_balance.agg.curr.feather: data/POS_CASH_balance.preprocessed.feather aggregate_pos_curr.py
	python aggregate_pos_curr.py

data/POS_CASH_balance.agg.prev.feather: data/POS_CASH_balance.preprocessed.feather aggregate_pos_prev.py
	python aggregate_pos_prev.py

# inst
INST := data/installments_payments.feather
INST_PREP := data/installments_payments.preprocessed.feather
INST_AGG_CURR := data/installments_payments.agg.curr.feather
INST_AGG_LAST := data/installments_payments.agg.curr.last.feather

.PHONY: inst
inst: $(INST_AGG_CURR) $(INST_AGG_LAST)
$(INST_PREP): $(INST) preprocess_inst.py
	python preprocess_inst.py
$(INST_AGG_CURR): $(INST_PREP) aggregate_inst_curr.py
	python aggregate_inst_curr.py
$(INST_AGG_LAST): $(INST_PREP) aggregate_inst_last.py
	python aggregate_inst_last.py

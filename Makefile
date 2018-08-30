default:
	echo "do nothing"

include ./feather.mk
include ./app.mk
include ./prev.mk
include ./bureau.mk
include ./pos.mk
include ./inst.mk
include ./credit.mk

# folds
FOLDS := $(shell seq 0 4)
FOLDS := $(addsuffix .feather,$(FOLDS))
FOLDS := $(addprefix data/fold.,$(FOLDS))

.PHONY: fold
fold: $(FOLDS)
$(FOLDS): data/application_train.feather split_fold.py
	python split_fold.py

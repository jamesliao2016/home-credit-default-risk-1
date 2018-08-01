# inst
INST := data/installments_payments.feather
INST_PREP := data/inst.preprocessed.feather

.PHONY: inst
inst: $(INST_AGG_CURR) $(INST_AGG_LAST)
$(INST_PREP): $(INST) preprocess_inst.py
	python preprocess_inst.py
data/inst.agg.feather: $(INST_PREP) aggregate_inst.py
	python aggregate_inst.py
data/inst.last.feather: $(INST_PREP) create_inst_last.py
	python create_inst_last.py
data/inst.tail.feather: $(INST_PREP) create_inst_tail.py
	python create_inst_tail.py
data/inst.diff.feather: $(INST_PREP) create_inst_diff.py
	python create_inst_diff.py

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
data/installments_payments.agg.prev.last.feather: data/installments_payments.preprocessed.feather aggregate_inst_prev_last.py
	python aggregate_inst_prev_last.py

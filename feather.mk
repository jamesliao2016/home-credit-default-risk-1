SRCS := $(shell ls data | grep csv.zip)
SRCS := $(addprefix data/,$(SRCS))
DSTS := $(SRCS:.csv.zip=.feather)

.PHONY: feather
feather: $(DSTS)

$(DSTS): $(SRCS) convert.py
	python convert.py --src $< --dst $@

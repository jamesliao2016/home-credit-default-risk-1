default:
	echo "do nothing"

SRCS := $(shell ls data | grep csv.zip)
SRCS := $(addprefix data/,$(SRCS))
DSTS := $(SRCS:.csv.zip=.feather)

.PHONY: all
all: $(DSTS)

$(DSTS): %.feather: %.csv.zip
	python convert.py --src $< --dst $@

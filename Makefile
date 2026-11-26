PYTHON ?= python

.PHONY: test reproduce demo

test:
	$(PYTHON) -m pytest -v

reproduce:
	$(PYTHON) -m babel.cli reproduce

demo:
	$(PYTHON) -m streamlit run src/babel/ui/app.py

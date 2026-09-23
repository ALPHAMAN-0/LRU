# Convenience wrappers. Everything here also works as a plain python3 command;
# see README.md. PYTHON can be overridden: make test PYTHON=/usr/bin/python3
PYTHON  ?= python3
PYTHONS ?= /usr/bin/python3 python3.9 python3.10 python3.11 python3.12 \
           python3.13 python3.14 $(PYTHON)

.PHONY: help demo test test-all shots verify clean

help:
	@echo "make demo      - run the demonstration (sections 1-3)"
	@echo "make test      - run the unit test suite"
	@echo "make test-all  - run the suite on each Python 3.9+ found (fails if any fails)"
	@echo "make shots     - regenerate photo/*.png from real runs"
	@echo "make verify    - tests + proof the captured logs match real output"
	@echo "make clean     - remove __pycache__ (never touches photo/)"

demo:
	$(PYTHON) demo.py

test:
	$(PYTHON) -m unittest -v

# Runs each interpreter in $(PYTHONS) that exists, deduplicated by version, and
# exits non-zero if ANY of them fails. Piping to `tail` would swallow the exit
# status, so each run is captured to a temp file instead.
test-all:
	@rc=0; seen=""; \
	for py in $(PYTHONS); do \
		command -v $$py >/dev/null 2>&1 || continue; \
		ver=$$($$py -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])' 2>/dev/null) || continue; \
		case " $$seen " in *" $$ver "*) continue;; esac; \
		seen="$$seen $$ver"; \
		echo "--- Python $$ver ---"; \
		log=$$(mktemp); \
		if $$py -m unittest >$$log 2>&1; then tail -3 $$log; \
		else rc=1; echo "  FAILED:"; tail -25 $$log; fi; \
		rm -f $$log; \
	done; \
	if [ $$rc -ne 0 ]; then echo; echo "At least one interpreter FAILED."; fi; \
	exit $$rc

shots:
	$(PYTHON) scripts/capture_screenshots.py

verify:
	@echo "== unit tests =="
	@log=$$(mktemp); \
	if $(PYTHON) -m unittest >$$log 2>&1; then tail -3 $$log; rm -f $$log; \
	else tail -30 $$log; rm -f $$log; echo; echo "TESTS FAILED"; exit 1; fi
	@echo
	@echo "== screenshot integrity =="
	@$(PYTHON) scripts/verify_photos.py
	@echo
	@echo "All checks passed."

clean:
	rm -rf .pytest_cache
	find . -name __pycache__ -type d -prune -not -path './photo/*' -exec rm -rf {} +

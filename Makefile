VODS   ?= $(HOME)/vods
OUT    ?= $(CURDIR)/tracksplit-out
DB     ?= $(OUT)/tracksplit.db

.PHONY: setup ingest dry run tui status reset

setup:
	uv venv && uv pip install -e .
	@echo "export ACR_HOST/ACR_KEY/ACR_SECRET (from your secret manager)"

ingest:      ## register every mp4 under $(VODS)
	uv run tracksplit ingest $(VODS) --db $(DB)

dry:         ## extract + gate only, print region map, no ACR, no cuts
	uv run tracksplit run --db $(DB) --out $(OUT) --stop-after gate

run:         ## full pipeline for everything pending
	uv run tracksplit run --db $(DB) --out $(OUT)

tui:         ## triage segments
	uv run tracksplit tui --out $(OUT)

status:
	uv run tracksplit status --db $(DB)

reset:       ## clear stage state but keep ACR cache
	uv run tracksplit reset --db $(DB)

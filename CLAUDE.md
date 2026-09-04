# tracksplit

DJ set VODs to per-song clips. Pipeline stages, setup and the Makefile targets are in `README.md`.

## Tests

```
uv run --with pytest pytest -q tests
```

Run them before every commit; each commit is one behaviour change with the test that pins it.

## Pull requests

Every PR body follows `.github/PULL_REQUEST_TEMPLATE.md`. The rules, the exceptions and the
redaction list are in `.github/PR_STANDARD.md`, and the **PR Standard** workflow enforces them.
In short: H1 and a one-paragraph lede, a Demo (GIF plus mp4, published with
`scripts/publish-demo.sh <PR> recording.mp4`, or one honest `**Demo waived:** <reason>` line),
a Test Plan with checked boxes for what actually ran, `## Proof (redacted)` with real output
collapsed in `<details>`, collapsed Full Summary and Files blocks, and a Provenance section.
Docs-only PRs skip the demo and proof automatically.

Redact before recording or pasting: VOD names and track titles become `<vod>` and
`<artist> - <title>`; hosts, home paths, serials, emails and keys never appear. Never commit
VODs, outputs or the ACR cache; ACR credentials come from the environment only.

Check a drafted body before opening the PR:

```
git diff --name-only origin/main...HEAD > /tmp/changed.txt
python3 .github/scripts/check_pr_standard.py --body /tmp/pr-body.md --files /tmp/changed.txt
```

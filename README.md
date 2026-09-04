# Paper Generation 2026

This repository develops three independent ICML-style research papers:

- `sft_paper`: supervised fine-tuning methods;
- `rl_paper`: reinforcement-learning methods; and
- `eval_paper`: evaluation methods and benchmarks.

The repository is currently at the methods-first drafting stage. Each paper
contains a research question, verified related-work bibliography, detailed
method, scoped formal analysis, and preregistered experimental protocol.
Empirical result tables are intentionally blank until the corresponding
experiments are executed; the drafts are not submission-ready results papers.

## Build

The templates use the unmodified official ICML 2026 LaTeX style in
`template/icml2026`. You need a TeX distribution with `pdflatex` (and
ideally `latexmk`). `make` prepends `/Library/TeX/texbin` on macOS and
falls back to `pdflatex` + `bibtex` if `latexmk` is not installed.

```bash
make
```

If `make` reports `latexmk: command not found` or `pdflatex not found`,
install TeX and refresh `PATH`:

```bash
# macOS, full MacTeX (includes latexmk)
brew install --cask mactex-no-gui
eval "$(/usr/libexec/path_helper)"
export PATH="/Library/TeX/texbin:$PATH"

# macOS, if MacTeX is already installed but the terminal cannot see it
export PATH="/Library/TeX/texbin:$PATH"

# Ubuntu/Debian
sudo apt-get install -y texlive-latex-extra texlive-fonts-recommended \
  texlive-science latexmk
```

Then retry `make`. Check which engine was found with `make check-deps`.

This builds:

```text
sft_paper/main.pdf
rl_paper/main.pdf
eval_paper/main.pdf
```

Run the CPU-only equation and protocol audits with:

```bash
make source-audit
make provenance-audit
make sft-factorial-audit
make protocol-audits
```

The source audit validates the committed pinned-metadata snapshot but leaves
all manual clearances pending. The provenance audit verifies row-level hashing,
normalized duplicate removal, cross-role quarantine, and output-manifest
integrity on a committed synthetic fixture. The factorial audit exercises the
registered hypervolume, paired difference-in-differences, missing-run, and
simultaneous max-$t$ aggregation rules without using empirical endpoints.
The ignored protocol artifacts are
synthetic software checks, not paper results. See `experiments/README.md` for
the five-stage execution contract and the boundary between these audits and
the planned model/human experiments.

Build or clean one paper with:

```bash
make sft_paper/main.pdf
make clean
```

## Experiment Protocols

Each methods-first draft has a machine-readable five-stage plan covering data
acquisition, processing, model construction, training, and evaluation. Every
plan also hashes its section of `experiments/pilot_preregistration.json`:

```bash
python3 scripts/run_experiment_pipeline.py \
  --plan experiments/sft_pipeline.json
python3 scripts/run_experiment_pipeline.py \
  --plan experiments/rl_pipeline.json
python3 scripts/run_experiment_pipeline.py \
  --plan experiments/eval_pipeline.json
```

These commands validate and display the plans. Site-local execution commands,
required artifacts, cost ledgers, and immutable output hashing are documented
in `experiments/README.md`. The repository does not claim that the pending
experiments have run.

## Authoring Rules

- Keep review submissions anonymous and leave `\usepackage{icml2026}` in
  review mode. Use the `accepted` option only for camera-ready papers.
- Keep the main body within eight pages. References, the impact statement,
  and appendices may follow without a page limit; camera-ready papers receive
  one additional main-body page.
- Put the mandatory Impact Statement before the references.
- Keep each abstract to one self-contained paragraph, ideally four to six
  sentences.
- Do not modify the ICML style files or compress spacing.
- Never replace `--` or `pending` result cells without an immutable run
  artifact and aggregation script.
- Add only verified citations and measured results. Never retain invented
  references, numbers, baselines, or implementation details.

See the official [ICML 2026 author instructions](https://icml.cc/Conferences/2026/AuthorInstructions)
and [call for papers](https://icml.cc/Conferences/2026/CallForPapers) before
submission.

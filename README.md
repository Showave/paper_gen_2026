# Paper Generation 2026

This repository develops three independent ICML-style research papers:

- `sft_paper`: supervised fine-tuning methods;
- `rl_paper`: reinforcement-learning methods; and
- `eval_paper`: evaluation methods and benchmarks.

The repository is currently at the first automation stage. It contains
formatting and paper-structure templates only; scientific claims, citations,
experiments, and results must be added and verified in later iterations.

## Build

The templates use the unmodified official ICML 2026 LaTeX style in
`template/icml2026`. A TeX Live installation with `latexmk` is required.

```bash
make
```

This builds:

```text
sft_paper/main.pdf
rl_paper/main.pdf
eval_paper/main.pdf
```

Build or clean one paper with:

```bash
make sft_paper/main.pdf
make clean
```

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
- Replace every visible `Template note` before treating a paper as a
  submission draft.
- Add only verified citations and measured results. Never retain invented
  references, numbers, baselines, or implementation details.

See the official [ICML 2026 author instructions](https://icml.cc/Conferences/2026/AuthorInstructions)
and [call for papers](https://icml.cc/Conferences/2026/CallForPapers) before
submission.

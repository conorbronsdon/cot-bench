# COT Bench paper

arXiv-ready methods paper accompanying the COT Bench leaderboard launch and the
Snorkel Open Benchmarks application. The benchmark is methodologically complete; the
Experiments section is a skeleton that populates from the first full evaluation run.

## Status

Drafted from the repo docs (`docs/methodology.md`, `docs/governance.md`,
`docs/scenario-schema.md`, `README.md`, `eval/config.py`) and the related-work
research reports. Every factual claim about the benchmark is taken from the code or
docs. Every headline number awaiting the eval run is a bracketed `\TODO{...}`
placeholder (red in the PDF); grep the sources for `\TODO` to find them all.

| Section | State |
|---------|-------|
| Abstract | Drafted; headline numbers bracketed |
| 1 Introduction | Drafted |
| 2 Benchmark Design | Drafted (corpus counts bracketed) |
| 3 Scoring | Drafted |
| 4 Statistical Methodology | Drafted |
| 5 Governance & Reproducibility | Drafted |
| 6 Experiments | Skeleton: every table/figure stub names its data source + generator |
| 7 Limitations | Drafted |
| 8 Related Work | Drafted with real citations |

## Layout

```
paper/
├── main.tex            # document root (article + common packages)
├── references.bib      # citations; each entry marked VERIFIED or TODO
├── sections/           # one .tex per section, \input from main.tex
├── figures/            # figures plan (README.md); generated PDFs land here
└── README.md           # this file
```

## Build

Self-contained; compiles with a standard TeX Live install (no vendored `.sty`).

```bash
cd paper
latexmk -pdf main.tex      # preferred; runs bibtex automatically
# or, manually:
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`latexmk -c` cleans aux files. To open in **Overleaf**, upload the `paper/` folder
(or zip it) and set `main.tex` as the main document; Overleaf runs the bibtex pass
automatically.

## Before submission

- Replace every `\TODO{...}` with the real value from the eval run.
- Regenerate the corpus counts in Table 1 with `scripts/validate_scenarios.py` and
  remove the placeholder brackets.
- Resolve the `TODO`-marked entries in `references.bib` (mostly future-dated arXiv
  ids that appear in the 2026-context research reports, plus a few author lists not
  individually confirmed).
- Build the `scripts/paper/` figure generators described in `figures/README.md`.
- Authorship/affiliation: independent / Chain of Thought. Do **not** position the
  author's Modular role as benchmark credibility (compliance boundary), and add no
  funding acknowledgment (the Snorkel application is not yet submitted).

## Citation note

The CLEAR framework is cited as Mehta, *Beyond Accuracy* (arXiv:2511.14136). The
repo `README.md` currently attributes CLEAR to "Simmering et al., 2025"; the arXiv
record lists Sushant Mehta. The README acknowledgment should be corrected to match.

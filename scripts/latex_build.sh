#!/bin/sh
# Build one paper directory (argument: sft_paper, rl_paper, or eval_paper).
# Prefers latexmk; falls back to pdflatex + bibtex. On macOS, prepends
# MacTeX's /Library/TeX/texbin so GUI TeX installs work from make.
set -eu

usage() {
  echo "usage: $0 <paper_dir> [build|clean|deps]" >&2
  exit 2
}

print_install_help() {
  cat >&2 <<'EOF'
error: no LaTeX engine found (need latexmk or pdflatex).

Install a TeX distribution, then open a new terminal and retry `make`.

  macOS (recommended):
    brew install --cask mactex-no-gui
    eval "$(/usr/libexec/path_helper)"
    export PATH="/Library/TeX/texbin:$PATH"

  macOS (smaller BasicTeX, then latexmk):
    brew install --cask basictex
    eval "$(/usr/libexec/path_helper)"
    sudo tlmgr update --self
    sudo tlmgr install latexmk collection-latexrecommended collection-fontsrecommended \
      natbib booktabs fancyhdr algorithms preprint

  Ubuntu/Debian:
    sudo apt-get update
    sudo apt-get install -y texlive-latex-recommended texlive-latex-extra \
      texlive-fonts-recommended texlive-science latexmk

If TeX is already installed, it is often missing from PATH. Try:
    export PATH="/Library/TeX/texbin:$PATH"
    make
EOF
}

have() {
  command -v "$1" >/dev/null 2>&1
}

if [ "$#" -lt 1 ]; then
  usage
fi

paper_dir=$1
mode=${2:-build}
repo_root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)

if [ -d /Library/TeX/texbin ]; then
  PATH="/Library/TeX/texbin:$PATH"
  export PATH
fi

# Linux TeX Live yeared prefix, if present and not already on PATH.
for texbin in /usr/local/texlive/*/bin/*; do
  if [ -d "$texbin" ]; then
    PATH="$texbin:$PATH"
    export PATH
  fi
done

if [ "$mode" = "deps" ]; then
  if have latexmk; then
    echo "latexmk: $(command -v latexmk)"
  else
    echo "latexmk: missing (make will fall back to pdflatex if available)"
  fi
  if have pdflatex; then
    echo "pdflatex: $(command -v pdflatex)"
  else
    echo "pdflatex: missing"
  fi
  if have bibtex; then
    echo "bibtex: $(command -v bibtex)"
  else
    echo "bibtex: missing"
  fi
  if have latexmk || have pdflatex; then
    exit 0
  fi
  print_install_help
  exit 127
fi

cd "$repo_root/$paper_dir"

export TEXINPUTS="../template/icml2026//:${TEXINPUTS:-}"
export BSTINPUTS="../template/icml2026//:${BSTINPUTS:-}"
export BIBINPUTS=".:${BIBINPUTS:-}"

if [ "$mode" = "clean" ]; then
  if have latexmk; then
    latexmk -C main.tex
  else
    rm -f main.aux main.bbl main.blg main.fdb_latexmk main.fls \
      main.log main.out main.pdf main.synctex.gz
  fi
  exit 0
fi

if [ "$mode" != "build" ]; then
  usage
fi

if have latexmk; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
  exit 0
fi

if ! have pdflatex; then
  print_install_help
  exit 127
fi

echo "warning: latexmk not found; using pdflatex + bibtex" >&2
pdflatex -interaction=nonstopmode -halt-on-error main.tex
if have bibtex; then
  bibtex main
else
  echo "warning: bibtex not found; bibliography may be incomplete" >&2
fi
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

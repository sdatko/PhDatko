.PHONY: all clean purge thesis

all: thesis

thesis: main.pdf

main.pdf: *.bib *.tex chapters/* images/*
	pdflatex --shell-escape main.tex -o main.pdf
	bibtex main.aux
	pdflatex --shell-escape main.tex -o main.pdf
	pdflatex --shell-escape main.tex -o main.pdf

clean:
	rm -f [a-zA-Z0-9]*.{aux,bbl,blg,lof,log,nav,out,snm,toc,vrb}
	rm -f chapters/[a-zA-Z0-9]*.{aux,bbl,blg,lof,log,nav,out,snm,toc,vrb}
	rm -rf _minted-[a-zA-Z0-9]*/

purge: clean
	rm -f main.pdf

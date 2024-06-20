
.PHONY: build
build:
	mkdir -p docs && cp -Rf images docs && cp support/styles.css docs && pandoc index.md -o docs/index.html --template=support/template.html --metadata title='Supervised Fine Tuning Mistral 7B' --metadata author='Dominic Kramer' --toc -s --mathjax

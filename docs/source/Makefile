# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXBUILD        = sphinx-build
SPHINXOPTS         = -j auto
SPHINXAUTOBUILD    = sphinx-autobuild
SPHINXAUTOOPTS     = -j auto
SOURCEDIR          = .
BUILDDIR           = ../build
HYPERACTIVEDIR     = ../../src/hyperactive

.PHONY: help build autobuild

# "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

clean:
	rm -rf $(BUILDDIR)
	rm -rf api_reference/auto_generated
	@echo "Deleted directory $(BUILDDIR) and auto_generated files."

# $(O) is meant as a shortcut for custom options.
# i.e to log stderr into a separate file:
# make build O="--no-color 2> build_warnings.log"
html:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# $(O) is meant as a shortcut for custom options.
autobuild:
	$(SPHINXAUTOBUILD) "$(SOURCEDIR)" "$(BUILDDIR)/html" -d "$(BUILDDIR)/doctrees" $(SPHINXAUTOOPTS) $(O) --watch "$(HYPERACTIVEDIR)" --re-ignore ".*\.json"

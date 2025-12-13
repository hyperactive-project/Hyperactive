# Hyperactive Documentation

This directory contains the Sphinx-based documentation for Hyperactive.

## Building the Documentation

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

You'll also need to have Hyperactive installed:

```bash
pip install -e ..  # Install Hyperactive in development mode from parent directory
```

### Building HTML Documentation

From the `source` directory:

```bash
cd source
make clean  # Clean previous builds
make html   # Build HTML documentation
```

The built documentation will be in `build/html/`. Open `build/html/index.html` in your browser to view.

### Live Preview with Auto-Rebuild

For development, you can use auto-rebuild mode:

```bash
cd source
make autobuild
```

This will start a local server (typically at http://127.0.0.1:8000) that automatically rebuilds when you make changes to the documentation source files.

## Documentation Structure

- `source/` - Documentation source files
  - `conf.py` - Sphinx configuration
  - `index.rst` - Main landing page
  - `api_reference/` - API reference documentation (auto-generated)
  - `user_guide/` - User guide pages (currently stubs)
  - `examples/` - Example notebooks and galleries (currently stubs)
  - `get_involved/` - Contributing guidelines (currently stubs)
  - `about/` - About pages (currently stubs)
  - `_templates/` - Custom Sphinx templates
  - `_static/` - Static files (CSS, images, etc.)
- `build/` - Built documentation (generated, not tracked in git)

## Current Status

The documentation is currently set up with:

- ✅ Full API reference auto-generated from docstrings
- ✅ Sphinx configuration following SK-Time's approach
- ✅ pydata_sphinx_theme for consistent look with scientific Python ecosystem
- ✅ Structural placeholders for future static content

Static pages (User Guide, Examples, etc.) are currently placeholder stubs marked "under construction" that can be filled in later.

## Adding New Content

### API Reference

The API reference is automatically generated from docstrings. To update:

1. Ensure your class/function has proper NumPy-style docstrings
2. Add the class/function to the appropriate `api_reference/*.rst` file using the `autosummary` directive
3. Rebuild the documentation

### Static Pages

To add content to the placeholder pages:

1. Edit the corresponding `.rst` or `.md` file in the appropriate directory
2. Remove the "under construction" note
3. Add your content using reStructuredText or Markdown syntax
4. Rebuild to see your changes

## Notes

- All API documentation is 100% auto-generated from source code docstrings
- The structure allows for easy addition of static content in the future
- Build warnings about missing references are normal during early development

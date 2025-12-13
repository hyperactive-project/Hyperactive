#!/usr/bin/env python3
"""
Badge Generator for Hyperactive Documentation

Generates local SVG badges based on project information from pyproject.toml.
This eliminates dependency on external badge services like shields.io.

Usage:
    python generate_badges.py

Output:
    - version.svg
    - python.svg
    - license.svg
    - sponsor.svg
"""

import re
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib


# Hyperactive color palette
COLORS = {
    "primary": "#5D5D7A",
    "secondary": "#7070A0",
    "dark": "#4A4A65",
    "light": "#f8f9fa",
    "sponsor": "#0eac92",
    "label_bg": "#555",
}


def create_badge_svg(label: str, value: str, color: str, label_width: int = None, value_width: int = None) -> str:
    """
    Create an SVG badge in shields.io flat-square style.

    Args:
        label: Left side text (e.g., "version")
        value: Right side text (e.g., "5.0.2")
        color: Hex color for the value background
        label_width: Override calculated label width
        value_width: Override calculated value width

    Returns:
        SVG string
    """
    # Approximate width calculation (7px per character + padding)
    char_width = 6.5
    padding = 10

    lw = label_width or int(len(label) * char_width + padding * 2)
    vw = value_width or int(len(value) * char_width + padding * 2)
    total_width = lw + vw
    height = 20

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="{height}">
  <title>{label}: {value}</title>
  <!-- Label background -->
  <rect width="{lw}" height="{height}" fill="{COLORS['label_bg']}"/>
  <!-- Value background -->
  <rect x="{lw}" width="{vw}" height="{height}" fill="{color}"/>
  <!-- Label text -->
  <text x="{lw/2}" y="14" fill="#fff" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11" text-anchor="middle">{label}</text>
  <!-- Value text -->
  <text x="{lw + vw/2}" y="14" fill="#fff" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11" text-anchor="middle">{value}</text>
</svg>'''

    return svg


def create_simple_badge_svg(text: str, color: str, width: int = None, font_size: int = 11) -> str:
    """
    Create a simple single-section SVG badge.

    Args:
        text: Badge text
        color: Background color
        width: Override calculated width
        font_size: Font size in pixels

    Returns:
        SVG string
    """
    char_width = 7
    padding = 16
    w = width or int(len(text) * char_width + padding * 2)
    height = 20
    text_y = 14 if font_size >= 11 else 13.5

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{height}">
  <title>{text}</title>
  <rect width="{w}" height="{height}" fill="{color}"/>
  <text x="{w/2}" y="{text_y}" fill="#fff" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="{font_size}" text-anchor="middle" font-weight="600">{text}</text>
</svg>'''

    return svg


def extract_python_versions(classifiers: list) -> str:
    """Extract Python version range from classifiers."""
    versions = []
    pattern = r"Programming Language :: Python :: (\d+\.\d+)"

    for classifier in classifiers:
        match = re.match(pattern, classifier)
        if match:
            versions.append(match.group(1))

    if not versions:
        return "3.10+"

    versions.sort(key=lambda v: tuple(map(int, v.split("."))))

    if len(versions) == 1:
        return versions[0]

    return f"{versions[0]} - {versions[-1]}"


def main():
    # Find pyproject.toml
    script_dir = Path(__file__).parent
    project_root = script_dir.parents[4]  # Go up from docs/source/_static/images/badges
    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        print(f"Error: pyproject.toml not found at {pyproject_path}")
        return

    # Read project info
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    project = pyproject.get("project", {})
    version = project.get("version", "0.0.0")
    classifiers = project.get("classifiers", [])

    # Extract license from classifiers
    license_name = "MIT"  # Default
    for classifier in classifiers:
        if "License :: OSI Approved ::" in classifier:
            license_name = classifier.split("::")[-1].strip()
            # Shorten common license names
            if "MIT" in license_name:
                license_name = "MIT"
            elif "BSD" in license_name:
                license_name = "BSD"
            elif "Apache" in license_name:
                license_name = "Apache 2.0"
            break

    python_range = extract_python_versions(classifiers)

    print(f"Generating badges for Hyperactive v{version}")
    print(f"  Python: {python_range}")
    print(f"  License: {license_name}")

    # Generate badges
    badges = {
        "version.svg": create_badge_svg("version", f"v{version}", COLORS["primary"]),
        "python.svg": create_badge_svg("python", python_range, COLORS["primary"]),
        "license.svg": create_badge_svg("license", license_name, COLORS["primary"]),
        "sponsor.svg": create_badge_svg("GC.OS", "Sponsored", COLORS["sponsor"]),
    }

    # Write badges
    for filename, svg_content in badges.items():
        filepath = script_dir / filename
        with open(filepath, "w") as f:
            f.write(svg_content)
        print(f"  Created: {filename}")

    print("\nDone! Badges generated successfully.")


if __name__ == "__main__":
    main()

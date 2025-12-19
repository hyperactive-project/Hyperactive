"""Example: Using Meta-Learn's feature extraction directly.

This example shows how to extract AST and search space features
using Meta-Learn's feature extraction module. These are the same
features that AlgorithmSelector uses internally.

Requirements:
    pip install meta-learn
"""

import math


def simple_quadratic(params):
    """Simple quadratic - unimodal, smooth."""
    return params["x"] ** 2 + params["y"] ** 2


def multimodal_sine(params):
    """Multimodal with trigonometry - many local optima."""
    x, y = params["x"], params["y"]
    return x**2 + y**2 + 10 * math.sin(x) * math.sin(y)


def conditional_function(params):
    """Discontinuous with conditionals."""
    x, y = params["x"], params["y"]
    if x > 0 and y > 0:
        return x**2 + y**2
    else:
        return abs(x) + abs(y) + 10


search_space = {
    "x": [i * 0.1 for i in range(-50, 51)],
    "y": [i * 0.1 for i in range(-50, 51)],
}


def main():
    try:
        from meta_learn.auto_opt.features import (
            extract_ast_features,
            extract_search_space_features,
        )
    except ImportError:
        print("Meta-Learn not available. Install with: pip install meta-learn")
        return

    functions = [
        ("Simple Quadratic", simple_quadratic),
        ("Multimodal Sine", multimodal_sine),
        ("Conditional", conditional_function),
    ]

    print("Feature Extraction with Meta-Learn")
    print("=" * 50)

    for name, func in functions:
        ast_feats = extract_ast_features(func)
        ss_feats = extract_search_space_features(search_space)

        print(f"\n{name}")
        print("-" * 40)

        print("AST Features (non-zero):")
        for k, v in ast_feats.to_dict().items():
            if v > 0:
                print(f"  {k}: {v}")

        print("Search Space Features:")
        for k, v in ss_feats.to_dict().items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

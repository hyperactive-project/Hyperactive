"""Consistency checks for integration wrappers.

Ensures that integration wrappers do not mix base-class ecosystems:
if a class uses skbase conventions (``_tags`` dict or ``get_test_params``),
it must also inherit from ``skbase.base.BaseObject``. Otherwise the
conventions are dead code and the class drops out of hyperactive's
registry-based test coverage.
"""

import importlib
import inspect
import pkgutil

from skbase.base import BaseObject

import hyperactive.integrations as _integrations_pkg


def _iter_integration_classes():
    """Yield classes defined under ``hyperactive.integrations``.

    Test modules and classes re-exported from other packages are skipped.
    Modules that fail to import (e.g. due to missing optional deps at
    collection time) are skipped as well so this test does not mask soft-dep
    handling done inside the modules themselves.
    """
    prefix = _integrations_pkg.__name__ + "."
    for mod_info in pkgutil.walk_packages(_integrations_pkg.__path__, prefix=prefix):
        name = mod_info.name
        if ".tests" in name or name.endswith(".tests"):
            continue
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == name:
                yield cls


def test_integrations_do_not_mix_base_class_ecosystems():
    """No Mischling wrappers: skbase conventions require skbase inheritance.

    A class that defines ``_tags`` as a dict or its own ``get_test_params``
    classmethod signals that it wants to participate in skbase-based
    registry and testing. Such a class must inherit from
    ``skbase.base.BaseObject``, otherwise the conventions are silently
    ignored.
    """
    offenders = []
    for cls in _iter_integration_classes():
        uses_skbase_conventions = (
            isinstance(cls.__dict__.get("_tags"), dict)
            or "get_test_params" in cls.__dict__
        )
        if uses_skbase_conventions and not issubclass(cls, BaseObject):
            offenders.append(f"{cls.__module__}.{cls.__name__}")

    assert not offenders, (
        "The following integration classes use skbase conventions (_tags "
        "or get_test_params) but do not inherit from skbase.base.BaseObject. "
        "Either inherit from BaseObject (or a skbase-based base class), or "
        "remove the skbase-style conventions:\n  - " + "\n  - ".join(sorted(offenders))
    )

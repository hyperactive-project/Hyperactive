# copyright: hyperactive developers, BSD-3-Clause License (see LICENSE file)
"""Testing of registry lookup functionality."""

# based on the sktime module of same name

__author__ = ["fkiraly"]

import pytest

from hyperactive.registry import all_objects

object_types = ["optimizer", "experiment"]


def _to_list(obj):
    """Put obj in list if it is not a list."""
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj.copy()


@pytest.mark.parametrize("return_names", [True, False])
@pytest.mark.parametrize("object_type", object_types)
def test_all_objects_by_scitype(object_type, return_names):
    """Check that all_objects return argument has correct type."""
    objects = all_objects(
        object_types=object_type,
        return_names=return_names,
    )

    assert isinstance(objects, list)
    # there should be at least one object returned
    assert len(objects) > 0

    # checks return type specification (see docstring)
    for obj in objects:
        if return_names:
            assert isinstance(obj, tuple) and len(obj) == 2
            name = obj[0]
            obj_cls = obj[1]
            assert isinstance(name, str)
            assert hasattr(obj_cls, "__name__")
            assert name == obj_cls.__name__
        else:
            obj_cls = obj

        assert hasattr(obj_cls, "get_tags")
        type_from_obj = obj_cls.get_class_tag("object_type")
        type_from_obj = _to_list(type_from_obj)

        assert object_type in type_from_obj

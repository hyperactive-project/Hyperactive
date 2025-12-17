"""Optuna adapter for search space conversion.

This module provides the adapter that converts SearchSpace objects
to the format expected by Optuna optimizers.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from __future__ import annotations

from typing import Any, Callable

from .._dimension import DimensionType
from .._search_space import SearchSpace
from ._base import BaseSearchSpaceAdapter

__all__ = ["OptunaSearchSpaceAdapter"]


class OptunaSearchSpaceAdapter(BaseSearchSpaceAdapter):
    """Adapter for Optuna.

    Optuna uses a define-by-run API where parameters are suggested
    through trial methods. This adapter creates appropriate distributions.

    Parameters
    ----------
    search_space : SearchSpace
        The search space to adapt.

    Examples
    --------
    >>> from hyperactive.search_space import SearchSpace
    >>> from hyperactive.search_space.adapters import OptunaSearchSpaceAdapter
    >>> space = SearchSpace(x=(0.0, 10.0), y=["a", "b", "c"])
    >>> adapter = OptunaSearchSpaceAdapter(space)
    >>> optuna_space = adapter.adapt()
    """

    def adapt(self, **kwargs) -> dict:
        """Convert search space to Optuna format.

        Returns a dictionary that can be used with Optuna's suggest methods
        or converted to distributions.

        Parameters
        ----------
        **kwargs
            Additional options (ignored).

        Returns
        -------
        dict
            Dictionary with parameter specifications for Optuna.
        """
        result = {}

        for name, dim in self.space.dimensions.items():
            result[name] = self._convert_dimension(dim)

        return result

    def _convert_dimension(self, dim) -> Any:
        """Convert a single dimension to Optuna format.

        Parameters
        ----------
        dim : Dimension
            The dimension to convert.

        Returns
        -------
        Any
            Optuna-compatible distribution or value specification.
        """
        if dim.dim_type == DimensionType.CATEGORICAL:
            # List for categorical choices
            return dim.values

        elif dim.dim_type == DimensionType.DISCRETE:
            # Convert numpy array to list for categorical
            return dim.values.tolist()

        elif dim.dim_type == DimensionType.CONTINUOUS:
            # Tuple for continuous range
            return (dim.low, dim.high)

        elif dim.dim_type == DimensionType.CONTINUOUS_LOG:
            # Return as Optuna FloatDistribution with log=True
            try:
                import optuna.distributions

                return optuna.distributions.FloatDistribution(
                    dim.low, dim.high, log=True
                )
            except ImportError:
                # Fallback: return tuple with marker
                return (dim.low, dim.high, "log")

        elif dim.dim_type == DimensionType.DISTRIBUTION:
            # Try to convert scipy to Optuna distribution
            return self._scipy_to_optuna(dim.values, dim)

        elif dim.dim_type == DimensionType.CONSTANT:
            # Single-element list for constant
            return [dim.values]

        else:
            raise ValueError(f"Unknown dimension type: {dim.dim_type}")

    def _scipy_to_optuna(self, dist, dim) -> Any:
        """Convert scipy distribution to Optuna distribution.

        Parameters
        ----------
        dist : scipy.stats.rv_frozen
            The scipy frozen distribution.
        dim : Dimension
            The dimension object.

        Returns
        -------
        Any
            Optuna distribution or sampled values.
        """
        try:
            import optuna.distributions

            # Check distribution type
            dist_name = getattr(dist, "dist", None)
            if dist_name is not None:
                name = dist_name.name

                # Handle common scipy distributions
                if name == "uniform":
                    loc, scale = dist.args[0] if dist.args else 0, dist.kwds.get(
                        "scale", 1
                    )
                    if not dist.args:
                        loc = dist.kwds.get("loc", 0)
                        scale = dist.kwds.get("scale", 1)
                    else:
                        loc = dist.args[0]
                        scale = dist.args[1] if len(dist.args) > 1 else 1
                    return optuna.distributions.FloatDistribution(loc, loc + scale)

                elif name == "loguniform":
                    # scipy.stats.loguniform(a, b) samples from [a, b] in log scale
                    low = dist.args[0] if dist.args else dist.kwds.get("a", 1e-5)
                    high = dist.args[1] if len(dist.args) > 1 else dist.kwds.get(
                        "b", 1e-1
                    )
                    return optuna.distributions.FloatDistribution(low, high, log=True)

                elif name == "randint":
                    low = dist.args[0] if dist.args else dist.kwds.get("low", 0)
                    high = dist.args[1] if len(dist.args) > 1 else dist.kwds.get(
                        "high", 10
                    )
                    return optuna.distributions.IntDistribution(low, high - 1)

            # Fallback: sample and use categorical
            samples = dist.rvs(size=100)
            return list(set(samples.tolist()))

        except ImportError:
            # Optuna not available, return sampled values
            samples = dist.rvs(size=100)
            return list(set(samples.tolist()))

    def create_suggest_function(self) -> Callable:
        """Create a define-by-run suggest function for conditional spaces.

        This function can be used in Optuna's objective function to
        suggest parameters with proper handling of conditional dimensions.

        Returns
        -------
        callable
            A function that takes an Optuna trial and returns suggested params.
        """

        def suggest(trial) -> dict:
            params = {}

            # Sort dimensions so dependencies come first
            sorted_dims = self._topological_sort_dimensions()

            for name in sorted_dims:
                dim = self.space.dimensions[name]

                # Check if this dimension is active
                if not self._is_dimension_active(name, params):
                    continue

                # Suggest based on dimension type
                params[name] = self._suggest_dimension(trial, name, dim)

            return params

        return suggest

    def _is_dimension_active(self, name: str, current_params: dict) -> bool:
        """Check if dimension should be suggested given current params.

        Parameters
        ----------
        name : str
            The dimension name.
        current_params : dict
            Currently sampled parameters.

        Returns
        -------
        bool
            True if the dimension should be suggested.
        """
        for condition in self.space.conditions:
            if condition.target_param == name:
                # Check if all dependencies are already sampled
                if condition.can_evaluate(current_params):
                    if not condition.is_active(current_params):
                        return False
        return True

    def _suggest_dimension(self, trial, name: str, dim) -> Any:
        """Suggest a value for a dimension using Optuna trial.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object.
        name : str
            The parameter name.
        dim : Dimension
            The dimension object.

        Returns
        -------
        Any
            The suggested value.
        """
        if dim.dim_type == DimensionType.CATEGORICAL:
            return trial.suggest_categorical(name, dim.values)

        elif dim.dim_type == DimensionType.DISCRETE:
            return trial.suggest_categorical(name, dim.values.tolist())

        elif dim.dim_type == DimensionType.CONTINUOUS:
            if dim.dtype == int:
                return trial.suggest_int(name, int(dim.low), int(dim.high))
            return trial.suggest_float(name, dim.low, dim.high)

        elif dim.dim_type == DimensionType.CONTINUOUS_LOG:
            return trial.suggest_float(name, dim.low, dim.high, log=True)

        elif dim.dim_type == DimensionType.DISTRIBUTION:
            # Sample from scipy and suggest categorical
            samples = dim.values.rvs(size=100)
            unique_samples = list(set(samples.tolist()))
            return trial.suggest_categorical(name, unique_samples)

        elif dim.dim_type == DimensionType.CONSTANT:
            return dim.values

        else:
            raise ValueError(f"Unknown dimension type: {dim.dim_type}")

    def _topological_sort_dimensions(self) -> list[str]:
        """Sort dimensions so dependencies come before dependents.

        Returns
        -------
        list[str]
            Sorted list of dimension names.
        """
        # Build dependency graph
        dependencies = {name: set() for name in self.space.dimensions}
        for condition in self.space.conditions:
            if condition.target_param in dependencies:
                for dep in condition.depends_on:
                    if dep in self.space.dimensions:
                        dependencies[condition.target_param].add(dep)

        # Topological sort (Kahn's algorithm)
        result = []
        no_deps = [n for n, deps in dependencies.items() if not deps]

        while no_deps:
            n = no_deps.pop(0)
            result.append(n)
            for name, deps in dependencies.items():
                if n in deps:
                    deps.remove(n)
                    if not deps and name not in result:
                        no_deps.append(name)

        # Add any remaining (cyclic dependencies or isolated)
        for name in self.space.dimensions:
            if name not in result:
                result.append(name)

        return result

    def get_constraints(self) -> list:
        """Get constraints as a list of callables.

        Note: Optuna doesn't natively support constraints.
        These need to be handled in the objective function.

        Returns
        -------
        list[callable]
            List of constraint predicates.
        """
        return [c.predicate for c in self.space.constraints]

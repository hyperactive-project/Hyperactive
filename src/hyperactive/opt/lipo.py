"""LIPO optimizer integration for Hyperactive."""

import numpy as np


class LIPOOptimizer:
    """Parameter-free global optimizer via the lipo package."""

    def __init__(self, search_space, n_iter, experiment, maximize=True):
        self.search_space = search_space
        self.n_iter = n_iter
        self.experiment = experiment
        self.maximize = maximize

    def _parse_search_space(self):
        lower, upper, cats = {}, {}, {}
        for key, values in self.search_space.items():
            # Categorical: list of strings
            if isinstance(values, list) and isinstance(values[0], str):
                cats[key] = values
            else:
                arr = np.array(values)
                lower[key] = float(arr.min())
                upper[key] = float(arr.max())
                # Store grid so we can snap results back later
                self._grids = getattr(self, "_grids", {})
                self._grids[key] = arr
        return lower, upper, cats

    def _snap_to_grid(self, params):
        """Snap lipo's continuous output to nearest valid grid point."""
        snapped = {}
        for key, val in params.items():
            if key in getattr(self, "_grids", {}):
                grid = self._grids[key]
                snapped[key] = grid[np.argmin(np.abs(grid - val))]
            else:
                snapped[key] = val  # categorical, pass through
        return snapped

    def solve(self):
        """Run optimizer and return best parameters as a dict."""
        from lipo import GlobalOptimizer

        lower, upper, cats = self._parse_search_space()

        def wrapped(**kwargs):
            return self.experiment(self._snap_to_grid(kwargs))

        opt = GlobalOptimizer(
            wrapped,
            lower_bounds=lower,
            upper_bounds=upper,
            categories=cats,
            maximize=self.maximize,
        )
        opt.run(self.n_iter)

        return self._snap_to_grid(opt.optimum[0])

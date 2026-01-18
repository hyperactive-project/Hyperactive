"""Utility functions for search space adaptation."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

from ._search_space_adapter import SearchSpaceAdapter

__all__ = ["adapt_search_space", "detect_search_space_key"]


def detect_search_space_key(search_config):
    """Find which key holds the search space in the config.

    Parameters
    ----------
    search_config : dict
        The search configuration dictionary.

    Returns
    -------
    str or None
        The key name for search space, or None if not found.
    """
    for key in ["search_space", "param_space", "param_grid", "param_distributions"]:
        if key in search_config and search_config[key] is not None:
            return key
    return None


def adapt_search_space(experiment, search_config, capabilities):
    """Adapt search space and experiment for backend capabilities.

    If the backend doesn't support certain search space features
    (e.g., categorical values, continuous ranges), this function:
    - Validates the search space format
    - Encodes categorical dimensions (strings to integers)
    - Discretizes continuous dimensions (tuples to lists)
    - Wraps the experiment to decode parameters during scoring

    Parameters
    ----------
    experiment : BaseExperiment
        The experiment to optimize.
    search_config : dict
        The search configuration containing the search space.
    capabilities : dict
        Backend capabilities, e.g., {"categorical": True, "continuous": False}.

    Returns
    -------
    experiment : BaseExperiment
        The experiment, possibly wrapped for decoding.
    search_config : dict
        The search config, possibly with encoded/discretized search space.
    adapter : SearchSpaceAdapter or None
        The adapter if adaptation was applied, None otherwise.

    Raises
    ------
    ValueError, TypeError
        If the search space format is invalid.
    """
    search_space_key = detect_search_space_key(search_config)

    # No search space found - pass through unchanged
    if not search_space_key or not search_config.get(search_space_key):
        return experiment, search_config, None

    # Create adapter with backend capabilities
    adapter = SearchSpaceAdapter(search_config[search_space_key], capabilities)

    # Validate search space format
    adapter.validate()

    # Backend supports all features - pass through unchanged
    if not adapter.needs_adaptation:
        return experiment, search_config, None

    # Adaptation needed - transform search space and wrap experiment
    encoded_config = search_config.copy()
    encoded_config[search_space_key] = adapter.encode()
    wrapped_experiment = adapter.wrap_experiment(experiment)

    return wrapped_experiment, encoded_config, adapter

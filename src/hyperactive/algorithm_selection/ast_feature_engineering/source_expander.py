"""Source code expander that follows function calls to collect full source.

This module provides functionality to recursively expand function calls
in an objective function, collecting the source code of all user-defined
functions that are called. It stops at external library boundaries.
"""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import ast
import importlib
import inspect
from typing import Callable, Optional

# Known external library prefixes - we stop expanding when we hit these
EXTERNAL_LIBRARIES = frozenset(
    {
        # Scientific computing
        "numpy",
        "np",
        "scipy",
        "pandas",
        "pd",
        # Machine learning
        "sklearn",
        "keras",
        "tensorflow",
        "tf",
        "torch",
        "xgboost",
        "lightgbm",
        "catboost",
        # Visualization
        "matplotlib",
        "plt",
        "seaborn",
        "sns",
        "plotly",
        # Image processing
        "cv2",
        "PIL",
        "skimage",
        # Other common libraries
        "requests",
        "json",
        "os",
        "sys",
        "re",
        "math",
        "random",
        "collections",
        "itertools",
        "functools",
        "typing",
    }
)


class SourceExpander:
    """Recursively expand function calls to collect full source code.

    This class takes an objective function and expands all user-defined
    function calls, collecting their source code. It stops at external
    library boundaries (numpy, sklearn, etc.) and handles:

    - Functions in __globals__ (same module)
    - Local imports inside functions
    - Nested function definitions

    Parameters
    ----------
    max_depth : int, default=10
        Maximum recursion depth for following function calls.
    external_libraries : frozenset, optional
        Set of library names to treat as external (stop expansion).
        Defaults to EXTERNAL_LIBRARIES.

    Attributes
    ----------
    collected_sources : list of str
        Source code strings collected during expansion.
    visited : set of int
        IDs of functions already visited (cycle detection).
    resolution_log : list of dict
        Log of resolution attempts for debugging.

    Examples
    --------
    >>> def helper(x):
    ...     return x ** 2
    >>> def objective(x):
    ...     return helper(x["val"])
    >>> expander = SourceExpander()
    >>> full_source = expander.expand(objective)
    >>> "helper" in full_source
    True
    """

    def __init__(
        self,
        max_depth: int = 10,
        external_libraries: Optional[frozenset] = None,
    ):
        self.max_depth = max_depth
        self.external_libraries = external_libraries or EXTERNAL_LIBRARIES

        # State reset on each expand() call
        self.collected_sources: list[str] = []
        self.visited: set[int] = set()
        self.resolution_log: list[dict] = []

    def expand(self, func: Callable) -> str:
        """Expand a function to include all called user-defined functions.

        Parameters
        ----------
        func : Callable
            The function to expand.

        Returns
        -------
        str
            Combined source code of the function and all user-defined
            functions it calls (transitively).
        """
        self._reset_state()
        self._expand_recursive(func, depth=0)
        return "\n\n".join(self.collected_sources)

    def _reset_state(self):
        """Reset internal state for a new expansion."""
        self.collected_sources = []
        self.visited = set()
        self.resolution_log = []

    def _expand_recursive(self, func: Callable, depth: int):
        """Recursively expand function calls.

        Parameters
        ----------
        func : Callable
            Function to expand.
        depth : int
            Current recursion depth.
        """
        # Stop conditions
        if depth > self.max_depth:
            self._log_resolution(func, "max_depth_exceeded", success=False)
            return

        if id(func) in self.visited:
            self._log_resolution(func, "already_visited", success=False)
            return

        if self._is_external(func):
            self._log_resolution(func, "external_library", success=False)
            return

        self.visited.add(id(func))

        # Get source code
        try:
            source = inspect.getsource(func)
            self.collected_sources.append(source)
            self._log_resolution(func, "source_collected", success=True)
        except (TypeError, OSError) as e:
            self._log_resolution(func, f"source_error: {e}", success=False)
            return

        # Parse and find function calls
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            self._log_resolution(func, f"parse_error: {e}", success=False)
            return

        # Find all function calls and nested definitions
        resolver = FunctionResolver(func, source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                callee = resolver.resolve_call(node)
                if callee is not None:
                    self._expand_recursive(callee, depth + 1)

    def _is_external(self, func: Callable) -> bool:
        """Check if a function is from an external library.

        Parameters
        ----------
        func : Callable
            Function to check.

        Returns
        -------
        bool
            True if the function is from an external library.
        """
        try:
            module = inspect.getmodule(func)
            if module is None:
                return True  # Can't determine, treat as external

            module_name = module.__name__.split(".")[0]

            # Check against known externals
            if module_name in self.external_libraries:
                return True

            # Check if it's a built-in
            if module_name in ("builtins", "__builtin__"):
                return True

            # Try to get source - C extensions will fail
            inspect.getsource(func)
            return False

        except (TypeError, OSError):
            return True  # Can't get source = external/built-in

    def _log_resolution(self, func: Callable, status: str, success: bool):
        """Log a resolution attempt for debugging.

        Parameters
        ----------
        func : Callable
            Function being resolved.
        status : str
            Status message.
        success : bool
            Whether resolution was successful.
        """
        try:
            name = getattr(func, "__name__", str(func))
            module = getattr(inspect.getmodule(func), "__name__", "unknown")
        except Exception:
            name = str(func)
            module = "unknown"

        self.resolution_log.append(
            {
                "function": name,
                "module": module,
                "status": status,
                "success": success,
            }
        )


class FunctionResolver:
    """Resolve function calls from AST nodes to actual function objects.

    This class handles multiple resolution strategies:
    1. Functions in __globals__ (module-level)
    2. Local imports inside the function
    3. Nested function definitions (returns source, not object)

    Parameters
    ----------
    func : Callable
        The parent function containing the calls.
    source : str
        Source code of the parent function.

    Examples
    --------
    >>> def objective(x):
    ...     return x ** 2
    >>> resolver = FunctionResolver(objective, inspect.getsource(objective))
    """

    def __init__(self, func: Callable, source: str):
        self.func = func
        self.source = source
        self._tree: Optional[ast.AST] = None
        self._local_imports: Optional[dict] = None
        self._nested_functions: Optional[dict] = None

    @property
    def tree(self) -> ast.AST:
        """Lazily parse the AST."""
        if self._tree is None:
            self._tree = ast.parse(self.source)
        return self._tree

    @property
    def local_imports(self) -> dict[str, tuple[str, str]]:
        """Lazily extract local imports.

        Returns dict mapping local name to (module_name, attr_name).
        """
        if self._local_imports is None:
            self._local_imports = self._extract_local_imports()
        return self._local_imports

    @property
    def nested_functions(self) -> dict[str, str]:
        """Lazily extract nested function definitions.

        Returns dict mapping function name to source code.
        """
        if self._nested_functions is None:
            self._nested_functions = self._extract_nested_functions()
        return self._nested_functions

    def resolve_call(self, call_node: ast.Call) -> Optional[Callable]:
        """Resolve an ast.Call node to an actual function.

        Parameters
        ----------
        call_node : ast.Call
            The AST call node to resolve.

        Returns
        -------
        Callable or None
            The resolved function, or None if resolution failed.
        """
        # Handle simple name: func()
        if isinstance(call_node.func, ast.Name):
            name = call_node.func.id
            return self._resolve_name(name)

        # Handle attribute access: module.func() or obj.method()
        if isinstance(call_node.func, ast.Attribute):
            return self._resolve_attribute(call_node.func)

        return None

    def _resolve_name(self, name: str) -> Optional[Callable]:
        """Resolve a simple function name.

        Tries in order:
        1. __globals__ (module-level functions/imports)
        2. Local imports inside the function
        3. Nested function definitions (not supported as callable)

        Parameters
        ----------
        name : str
            Function name to resolve.

        Returns
        -------
        Callable or None
            The resolved function, or None if not found.
        """
        # Strategy 1: Check __globals__
        if hasattr(self.func, "__globals__") and name in self.func.__globals__:
            obj = self.func.__globals__[name]
            if callable(obj):
                return obj

        # Strategy 2: Check local imports
        if name in self.local_imports:
            module_name, attr_name = self.local_imports[name]
            resolved = self._import_and_resolve(module_name, attr_name)
            if resolved is not None:
                return resolved

        # Strategy 3: Nested functions - these are in the source already
        # We return None because the source is already collected
        # when we parse the parent function
        if name in self.nested_functions:
            return None

        return None

    def _resolve_attribute(self, attr_node: ast.Attribute) -> Optional[Callable]:
        """Resolve an attribute access like module.func or obj.method.

        Parameters
        ----------
        attr_node : ast.Attribute
            The AST attribute node.

        Returns
        -------
        Callable or None
            The resolved function, or None if resolution failed.
        """
        # Handle chain: get the base name
        parts = []
        node = attr_node

        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value

        if isinstance(node, ast.Name):
            parts.append(node.id)

        parts.reverse()

        if not parts:
            return None

        base_name = parts[0]
        attr_path = parts[1:]

        # Try to resolve from globals
        if hasattr(self.func, "__globals__") and base_name in self.func.__globals__:
            obj = self.func.__globals__[base_name]
            for attr in attr_path:
                try:
                    obj = getattr(obj, attr)
                except AttributeError:
                    return None
            if callable(obj):
                return obj

        return None

    def _extract_local_imports(self) -> dict[str, tuple[str, str]]:
        """Extract imports defined inside the function body.

        Returns
        -------
        dict
            Mapping from local name to (module_name, attribute_name).
            For `from x import y as z`, returns {"z": ("x", "y")}.
            For `import x as y`, returns {"y": ("x", None)}.
        """
        imports = {}

        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    local_name = alias.asname or alias.name
                    imports[local_name] = (module, alias.name)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    local_name = alias.asname or alias.name
                    imports[local_name] = (alias.name, None)

        return imports

    def _extract_nested_functions(self) -> dict[str, str]:
        """Extract nested function definitions.

        Returns
        -------
        dict
            Mapping from function name to source code.
        """
        nested = {}

        # Find the main function definition
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                # Look for nested FunctionDefs inside this one
                for child in ast.walk(node):
                    if isinstance(child, ast.FunctionDef) and child is not node:
                        try:
                            nested[child.name] = ast.unparse(child)
                        except Exception:
                            pass  # ast.unparse might fail in some edge cases

        return nested

    def _import_and_resolve(
        self, module_name: str, attr_name: Optional[str]
    ) -> Optional[Callable]:
        """Dynamically import a module and resolve an attribute.

        Parameters
        ----------
        module_name : str
            Name of the module to import.
        attr_name : str or None
            Attribute to get from the module. If None, returns the module.

        Returns
        -------
        Callable or None
            The resolved object, or None if import failed.
        """
        try:
            module = importlib.import_module(module_name)
            if attr_name is None:
                return module if callable(module) else None
            obj = getattr(module, attr_name)
            return obj if callable(obj) else None
        except (ImportError, AttributeError):
            return None


def get_expanded_source(func: Callable, max_depth: int = 10) -> str:
    """Convenience function to expand a function's source code.

    Parameters
    ----------
    func : Callable
        The function to expand.
    max_depth : int, default=10
        Maximum recursion depth.

    Returns
    -------
    str
        Combined source code of the function and all user-defined
        functions it calls.

    Examples
    --------
    >>> def helper(x):  # doctest: +SKIP
    ...     return x ** 2
    >>> def objective(x):  # doctest: +SKIP
    ...     return helper(x["val"])
    >>> source = get_expanded_source(objective)  # doctest: +SKIP
    >>> "helper" in source  # doctest: +SKIP
    True
    """
    expander = SourceExpander(max_depth=max_depth)
    return expander.expand(func)

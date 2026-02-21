from typing import Dict, Any, Optional, Tuple
import pandas as pd
from catboost import Pool, cv
from hyperactive.base._experiment import BaseExperiment

class CatBoostCvExperiment(BaseExperiment):
    """Cross-validation experiment for CatBoost using `catboost.cv()`.

    Wraps CatBoost cross-validation so it can be used directly with Hyperactive optimizers.
    Returns the final mean test metric over folds (e.g. test-Logloss-mean, test-AUC-mean).

    Parameters
    ----------
    pool : catboost.Pool
        Data pool with features, labels, and optional weights/groups.
    iterations : int, default=100
        Maximum boosting iterations.
    fold_count : int, default=5
        Number of cross-validation folds.
    early_stopping_rounds : int or None, default=None
    partition_random_seed : int, default=0
    type : str, default="Classical"
        CV scheme ('Classical', 'TimeSeries', 'Inverted', etc.).
    metric : str, default="Logloss"
        Metric to extract as the optimization score.
    loss_function : str, default="Logloss"
        Training objective. If different from `metric`, added as `custom_metric`.
    """

    _tags = {
        "object_type": "experiment",
        "python_dependencies": "catboost",
        "property:randomness": "random",
        "property:higher_or_lower_is_better": "lower"
    }

    def __init__(
        self,
        pool: Pool,
        iterations: int = 100,
        fold_count: int = 5,
        early_stopping_rounds: Optional[int] = None,
        partition_random_seed: int = 0,
        type: str = "Classical",
        metric: str = "Logloss",
        loss_function: str = "Logloss",
    ):
        super().__init__()
        self.pool = pool
        self.iterations = iterations
        self.fold_count = fold_count
        self.early_stopping_rounds = early_stopping_rounds
        self.partition_random_seed = partition_random_seed
        self.type = type
        self.metric = metric
        self.loss_function = loss_function

        direction = "lower"
        self.set_tags(**{"property:higher_or_lower_is_better": direction})

    def _paramnames(self) -> Optional[list]:
        return None  

    def _evaluate(self, params: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Run CatBoost CV and return mean test metric + metadata."""
        cv_params = params.copy()

        cv_params.setdefault("loss_function", self.loss_function)

        cv_params.setdefault("iterations", self.iterations)

        if self.metric != self.loss_function:
            custom_metrics = cv_params.get("custom_metric", [])
            if isinstance(custom_metrics, str):
                custom_metrics = [custom_metrics]
            if self.metric not in custom_metrics:
                custom_metrics.append(self.metric)
            cv_params["custom_metric"] = custom_metrics

        try:
            cv_results: pd.DataFrame = cv(
                params=cv_params,
                pool=self.pool,
                fold_count=self.fold_count,
                early_stopping_rounds=self.early_stopping_rounds,
                partition_random_seed=self.partition_random_seed,
                type=self.type,
                verbose=False,
                plot=False,
                return_models=False,
                as_pandas=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"CatBoost CV failed with params: {cv_params}\n"
                f"Error: {str(e)}"
            ) from e

        target_col = f"test-{self.metric}-mean"

        if target_col not in cv_results.columns:
            available = ", ".join(cv_results.columns)
            raise ValueError(
                f"Expected column '{target_col}' not found in cv_results.\n"
                f"Available columns: {available}\n"
                f"Check that metric='{self.metric}' is computed. "
                f"Current loss_function: '{cv_params.get('loss_function')}'"
            )

        mean_score = float(cv_results[target_col].iloc[-1])

        metadata = {
            "cv_results": cv_results.to_dict(orient="records"),
            "final_iteration": int(cv_results["iterations"].iloc[-1]),
            "target_column": target_col,
            "used_loss_function": cv_params["loss_function"],
            "custom_metrics_used": cv_params.get("custom_metric", None),
        }

        return mean_score, metadata
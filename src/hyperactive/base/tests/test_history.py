"""Tests for SearchHistory and history tracking in experiments."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import pytest

from hyperactive.base import SearchHistory


class TestSearchHistory:
    """Tests for the SearchHistory class."""

    def test_init_empty(self):
        """Test that a new SearchHistory is empty."""
        history = SearchHistory()
        assert history.n_trials == 0
        assert history.n_runs == 1
        assert history.history == []
        assert history.best_trial is None
        assert history.best_score is None
        assert history.best_params is None

    def test_record_single_trial(self):
        """Test recording a single trial."""
        history = SearchHistory()
        history.record(
            params={"x": 1, "y": 2},
            score=0.5,
            metadata={"time": 1.0},
            eval_time=0.1,
        )

        assert history.n_trials == 1
        assert history.n_runs == 1

        trial = history.history[0]
        assert trial["iteration"] == 0
        assert trial["run_id"] == 0
        assert trial["params"] == {"x": 1, "y": 2}
        assert trial["score"] == 0.5
        assert trial["metadata"] == {"time": 1.0}
        assert trial["eval_time"] == 0.1

    def test_record_multiple_trials(self):
        """Test recording multiple trials in one run."""
        history = SearchHistory()

        for i in range(5):
            history.record(
                params={"x": i},
                score=float(i),
                metadata={},
                eval_time=0.1,
            )

        assert history.n_trials == 5
        assert history.n_runs == 1

        # Check iteration is global
        for i, trial in enumerate(history.history):
            assert trial["iteration"] == i
            assert trial["run_id"] == 0

    def test_multiple_runs(self):
        """Test that run_id increments across multiple runs."""
        history = SearchHistory()

        history.record(params={"x": 1}, score=0.1, metadata={}, eval_time=0.1)
        history.record(params={"x": 2}, score=0.2, metadata={}, eval_time=0.1)

        history.new_run()
        history.record(params={"x": 3}, score=0.3, metadata={}, eval_time=0.1)

        assert history.n_trials == 3
        assert history.n_runs == 2

        # Check run_ids
        assert history.history[0]["run_id"] == 0
        assert history.history[1]["run_id"] == 0
        assert history.history[2]["run_id"] == 1

        # Iteration is global
        assert history.history[0]["iteration"] == 0
        assert history.history[1]["iteration"] == 1
        assert history.history[2]["iteration"] == 2

    def test_best_trial(self):
        """Test that best_trial returns the trial with highest score."""
        history = SearchHistory()
        history.record(params={"x": 1}, score=0.5, metadata={}, eval_time=0.1)
        history.record(params={"x": 2}, score=0.9, metadata={}, eval_time=0.1)
        history.record(params={"x": 3}, score=0.3, metadata={}, eval_time=0.1)

        best = history.best_trial
        assert best["score"] == 0.9
        assert best["params"] == {"x": 2}
        assert history.best_score == 0.9
        assert history.best_params == {"x": 2}

    def test_get_run(self):
        """Test filtering trials by run_id."""
        history = SearchHistory()

        history.record(params={"x": 1}, score=0.1, metadata={}, eval_time=0.1)
        history.record(params={"x": 2}, score=0.2, metadata={}, eval_time=0.1)

        history.new_run()
        history.record(params={"x": 3}, score=0.3, metadata={}, eval_time=0.1)

        run0 = history.get_run(0)
        run1 = history.get_run(1)

        assert len(run0) == 2
        assert len(run1) == 1
        assert run0[0]["params"] == {"x": 1}
        assert run0[1]["params"] == {"x": 2}
        assert run1[0]["params"] == {"x": 3}

    def test_clear(self):
        """Test that clear resets all history."""
        history = SearchHistory()
        history.record(params={"x": 1}, score=0.5, metadata={}, eval_time=0.1)

        history.clear()

        assert history.n_trials == 0
        assert history.n_runs == 1
        assert history.history == []

    def test_params_are_copied(self):
        """Test that recorded params are copied, not referenced."""
        history = SearchHistory()
        params = {"x": 1}
        history.record(params=params, score=0.5, metadata={}, eval_time=0.1)

        # Modify original
        params["x"] = 999

        # Recorded params should be unchanged
        assert history.history[0]["params"]["x"] == 1

    def test_metadata_none_becomes_empty_dict(self):
        """Test that None metadata becomes an empty dict."""
        history = SearchHistory()
        history.record(params={"x": 1}, score=0.5, metadata=None, eval_time=0.1)

        assert history.history[0]["metadata"] == {}

    def test_len(self):
        """Test __len__ returns number of trials."""
        history = SearchHistory()
        assert len(history) == 0

        history.record(params={"x": 1}, score=0.5, metadata={}, eval_time=0.1)
        assert len(history) == 1

    def test_repr(self):
        """Test __repr__ is informative."""
        history = SearchHistory()
        history.record(params={"x": 1}, score=0.5, metadata={}, eval_time=0.1)

        repr_str = repr(history)
        assert "n_trials=1" in repr_str
        assert "n_runs=1" in repr_str


class TestExperimentDataIntegration:
    """Tests for data tracking in BaseExperiment via accessor pattern."""

    def test_experiment_has_data_accessor(self):
        """Test that BaseExperiment has data accessor."""
        from hyperactive.base import SearchHistory
        from hyperactive.experiment.func import FunctionExperiment

        def objective(params):
            return params["x"] ** 2

        exp = FunctionExperiment(objective)

        assert hasattr(exp, "data")
        assert isinstance(exp.data, SearchHistory)
        assert exp.data.history == []
        assert exp.data.n_trials == 0

    def test_evaluate_records_data(self):
        """Test that evaluate() records trials to data."""
        from hyperactive.experiment.func import FunctionExperiment

        def objective(params):
            return params["x"] ** 2

        exp = FunctionExperiment(objective)

        exp.evaluate({"x": 2})
        exp.evaluate({"x": 3})

        assert exp.data.n_trials == 2
        assert len(exp.data.history) == 2

        trial0 = exp.data.history[0]
        assert trial0["params"] == {"x": 2}
        assert trial0["score"] == 4.0
        assert trial0["iteration"] == 0
        assert trial0["run_id"] == 0
        assert "eval_time" in trial0

    def test_score_records_via_evaluate(self):
        """Test that score() also records data (via evaluate)."""
        from hyperactive.experiment.func import FunctionExperiment

        def objective(params):
            return params["x"] ** 2

        exp = FunctionExperiment(objective)

        exp.score({"x": 5})

        assert exp.data.n_trials == 1
        assert exp.data.history[0]["score"] == 25.0

    def test_best_trial_property(self):
        """Test best_trial property via accessor."""
        from hyperactive.experiment.func import FunctionExperiment

        def objective(params):
            return params["x"]

        exp = FunctionExperiment(objective)

        exp.evaluate({"x": 1})
        exp.evaluate({"x": 5})
        exp.evaluate({"x": 3})

        assert exp.data.best_trial["score"] == 5.0
        assert exp.data.best_score == 5.0

    def test_clear_data(self):
        """Test data.clear() resets experiment data."""
        from hyperactive.experiment.func import FunctionExperiment

        def objective(params):
            return params["x"]

        exp = FunctionExperiment(objective)
        exp.evaluate({"x": 1})

        exp.data.clear()

        assert exp.data.n_trials == 0
        assert exp.data.history == []

    def test_get_run(self):
        """Test data.get_run() filters by run."""
        from hyperactive.experiment.func import FunctionExperiment

        def objective(params):
            return params["x"]

        exp = FunctionExperiment(objective)

        exp.evaluate({"x": 1})

        exp.data.new_run()
        exp.evaluate({"x": 2})

        run0 = exp.data.get_run(0)
        run1 = exp.data.get_run(1)

        assert len(run0) == 1
        assert len(run1) == 1
        assert run0[0]["params"] == {"x": 1}
        assert run1[0]["params"] == {"x": 2}


class TestOptimizerDataIntegration:
    """Tests for data tracking with optimizers."""

    def test_optimizer_records_trials(self):
        """Test that optimizer.solve() records trials to experiment data."""
        from hyperactive.experiment.func import FunctionExperiment
        from hyperactive.opt import RandomSearch

        def objective(params):
            return -((params["x"] - 2) ** 2)

        exp = FunctionExperiment(objective)
        opt = RandomSearch(
            experiment=exp,
            search_space={"x": [0, 1, 2, 3, 4]},
            n_iter=5,
        )

        opt.solve()

        assert exp.data.n_trials > 0
        assert all(t["run_id"] == 0 for t in exp.data.history)

    def test_multiple_solves_accumulate(self):
        """Test that multiple solve() calls accumulate trials."""
        from hyperactive.experiment.func import FunctionExperiment
        from hyperactive.opt import RandomSearch

        def objective(params):
            return -((params["x"] - 2) ** 2)

        exp = FunctionExperiment(objective)
        opt = RandomSearch(
            experiment=exp,
            search_space={"x": [0, 1, 2, 3, 4]},
            n_iter=3,
        )

        opt.solve()
        n_trials_first = exp.data.n_trials

        opt.solve()

        assert exp.data.n_trials > n_trials_first
        iterations = [t["iteration"] for t in exp.data.history]
        assert iterations == list(range(len(iterations)))

    def test_data_accumulates_different_optimizers(self):
        """Test data accumulates when using different optimizers."""
        from hyperactive.experiment.func import FunctionExperiment
        from hyperactive.opt import GridSearch, RandomSearch

        def objective(params):
            return -((params["x"] - 2) ** 2)

        exp = FunctionExperiment(objective)

        opt1 = RandomSearch(
            experiment=exp,
            search_space={"x": [0, 1, 2, 3, 4]},
            n_iter=3,
        )
        opt1.solve()
        n_trials_after_opt1 = exp.data.n_trials

        opt2 = GridSearch(
            experiment=exp,
            search_space={"x": [0, 1, 2, 3, 4]},
        )
        opt2.solve()

        assert exp.data.n_trials > n_trials_after_opt1
        iterations = [t["iteration"] for t in exp.data.history]
        assert iterations == list(range(len(iterations)))

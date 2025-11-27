"""Tests for callback functionality."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

import numpy as np

def test_experiment_callbacks_post():
    """Test that post-evaluation callbacks are called."""
    from hyperactive.experiment.bench import Sphere

    callback_calls = []

    def track_callback(exp, params, result, metadata):
        callback_calls.append(
            {
                "params": params.copy(),
                "result": result,
            }
        )

    exp = Sphere(d=2)
    exp.set_config(callbacks_post=[track_callback])

    exp.evaluate({"x0": 0.0, "x1": 0.0})
    exp.evaluate({"x0": 1.0, "x1": 1.0})
    exp.evaluate({"x0": 2.0, "x1": 2.0})

    assert len(callback_calls) == 3
    assert callback_calls[0]["params"] == {"x0": 0.0, "x1": 0.0}
    assert callback_calls[1]["params"] == {"x0": 1.0, "x1": 1.0}
    assert callback_calls[2]["params"] == {"x0": 2.0, "x1": 2.0}

def test_experiment_callbacks_pre():
    """Test that pre-evaluation callbacks are called."""
    from hyperactive.experiment.bench import Sphere

    pre_calls = []

    def pre_callback(exp, params):
        pre_calls.append(params.copy())

    exp = Sphere(d=2)
    exp.set_config(callbacks_pre=[pre_callback])

    exp.evaluate({"x0": 1.0, "x1": 2.0})

    assert len(pre_calls) == 1
    assert pre_calls[0] == {"x0": 1.0, "x1": 2.0}

def test_optimizer_callbacks():
    """Test optimizer pre/post solve callbacks."""
    from hyperactive.experiment.bench import Sphere
    from hyperactive.opt import HillClimbing

    pre_solve_called = []
    post_solve_called = []

    def pre_solve_cb(optimizer):
        pre_solve_called.append(True)

    def post_solve_cb(optimizer, best_params):
        post_solve_called.append(best_params)

    exp = Sphere(d=2)
    optimizer = HillClimbing(
        search_space={
            "x0": np.linspace(-5, 5, 11),
            "x1": np.linspace(-5, 5, 11),
        },
        n_iter=10,
        experiment=exp,
    )
    optimizer.set_config(
        callbacks_pre_solve=[pre_solve_cb],
        callbacks_post_solve=[post_solve_cb],
    )

    best_params = optimizer.solve()

    assert len(pre_solve_called) == 1
    assert len(post_solve_called) == 1
    assert post_solve_called[0] == best_params

def test_history_callback():
    """Test HistoryCallback records evaluations."""
    from hyperactive.experiment.bench import Sphere
    from hyperactive.opt import HillClimbing
    from hyperactive.utils.callbacks import HistoryCallback

    history_cb = HistoryCallback()

    exp = Sphere(d=2)
    exp.set_config(callbacks_post=[history_cb])

    optimizer = HillClimbing(
        search_space={
            "x0": np.linspace(-5, 5, 11),
            "x1": np.linspace(-5, 5, 11),
        },
        n_iter=10,
        experiment=exp,
    )

    optimizer.solve()

    assert len(history_cb.history) >= 10
    for record in history_cb.history:
        assert "params" in record
        assert "result" in record
        assert "metadata" in record

    best = history_cb.get_best(higher_is_better=False)
    assert best is not None
    assert "result" in best

def test_logging_callback(capsys):
    """Test LoggingCallback prints to stdout."""
    from hyperactive.experiment.bench import Sphere
    from hyperactive.utils.callbacks import LoggingCallback

    log_cb = LoggingCallback()

    exp = Sphere(d=2)
    exp.set_config(callbacks_post=[log_cb])

    exp.evaluate({"x0": 0.0, "x1": 0.0})
    exp.evaluate({"x0": 1.0, "x1": 1.0})

    captured = capsys.readouterr()
    assert "Eval 1:" in captured.out
    assert "Eval 2:" in captured.out

def test_sleep_callback():
    """Test SleepCallback adds delay."""
    import time

    from hyperactive.experiment.bench import Sphere
    from hyperactive.utils.callbacks import SleepCallback

    sleep_cb = SleepCallback(seconds=0.1)

    exp = Sphere(d=2)
    exp.set_config(callbacks_post=[sleep_cb])

    start = time.time()
    exp.evaluate({"x0": 0.0, "x1": 0.0})
    exp.evaluate({"x0": 1.0, "x1": 1.0})
    elapsed = time.time() - start

    assert elapsed >= 0.2

def test_target_reached_callback():
    """Test TargetReachedCallback tracks target."""
    from hyperactive.experiment.bench import Sphere
    from hyperactive.utils.callbacks import TargetReachedCallback

    target_cb = TargetReachedCallback(target_score=0.5, higher_is_better=False)

    exp = Sphere(d=2)
    exp.set_config(callbacks_post=[target_cb])

    exp.evaluate({"x0": 5.0, "x1": 5.0})
    assert not target_cb.reached

    exp.evaluate({"x0": 0.0, "x1": 0.0})
    assert target_cb.reached

def test_multiple_callbacks():
    """Test that multiple callbacks can be registered."""
    from hyperactive.experiment.bench import Sphere
    from hyperactive.utils.callbacks import HistoryCallback, LoggingCallback

    history_cb = HistoryCallback()
    log_cb = LoggingCallback()

    exp = Sphere(d=2)
    exp.set_config(callbacks_post=[history_cb, log_cb])

    exp.evaluate({"x0": 1.0, "x1": 1.0})
    exp.evaluate({"x0": 2.0, "x1": 2.0})

    assert len(history_cb.history) == 2
    assert log_cb._count == 2

import copy
import io
import os
from pathlib import Path
import re
import numpy as np

import pandas as pd
import pytest
import hashlib
from matplotlib import pyplot as plt
import matplotlib.testing.compare as plt_compare

from pathlib import Path
from ablator.analysis.plot.main import PlotAnalysis

from ablator import PlotAnalysis
from ablator.analysis.plot import Plot
from ablator.analysis.plot.cat_plot import Categorical, ViolinPlot
from ablator.analysis.results import Results
from ablator.analysis.plot.utils import parse_name_remap
from ablator.config.proto import ModelConfig, Optim, TrainConfig
from ablator.analysis.plot.num_plot import LinearPlot
from ablator.analysis.main import Analysis
from ablator.config.mp import ParallelConfig, SearchSpace
from ablator.config.proto import RunConfig
from ablator.config.main import ConfigBase
from ablator.modules.optimizer import OptimizerConfig
from ablator.modules.scheduler import SchedulerConfig


@pytest.mark.order(1)
def test_analysis(tmp_path: Path, ablator_results):
    results: Results = ablator_results
    PlotAnalysis(results, optim_metrics={"val_loss": "min"})
    categorical_name_remap = {
        "model_config.mock_param": "Some Parameter",
    }
    numerical_name_remap = {
        "train_config.optimizer_config.arguments.lr": "Learning Rate",
    }
    analysis = PlotAnalysis(
        results,
        save_dir=tmp_path.as_posix(),
        cache=True,
        optim_metrics={"val_loss": "min"},
    )
    attribute_name_remap = {**categorical_name_remap, **numerical_name_remap}
    analysis.make_figures(
        metric_name_remap={
            "val_loss": "Val. Loss",
        },
        attribute_name_remap=attribute_name_remap,
    )
    assert all(
        tmp_path.joinpath("violinplot", "val_loss", f"{file_name}.png").exists()
        for file_name in categorical_name_remap
    )

    assert all(
        tmp_path.joinpath("linearplot", "val_loss", f"{file_name}.png").exists()
        for file_name in numerical_name_remap
    )


def test_name_remap(tmp_path: Path):
    name_map = {"a": "A", "b": "B"}
    defaults = None
    assert parse_name_remap(defaults, name_map) == name_map

    name_map = {"a": "A", "b": "B"}
    defaults = ["a", "b", "c", "d"]
    expected_output = {"a": "A", "b": "B", "c": "c", "d": "d"}
    assert parse_name_remap(defaults, name_map) == expected_output

    name_map = None
    defaults = ["a", "b", "c"]
    assert parse_name_remap(defaults, name_map) == {"a": "a", "b": "b", "c": "c"}

    try:
        parse_name_remap(None, None)
    except NotImplementedError as err:
        if str(err) != "`defaults` or `name_map` argument required.":
            raise err


def test_analysis_func(tmp_path: Path):
    results_csv = Path(__file__).parent.parent.joinpath("assets", "results.csv")
    df = pd.read_csv(results_csv)

    categorical_name_remap = {
        "model_config.activation": "Activation",
        "model_config.initialization": "Weight Init.",
        "train_config.optimizer_config.name": "Optimizer",
        "model_config.mask_type": "Mask Type",
        "train_config.cat_nan_policy": "Policy for Cat. Missing",
        "train_config.normalization": "Dataset Normalization",
    }
    numerical_name_remap = {
        "model_config.n_heads": "N. Heads",
        "model_config.n_layers": "N. Layers",
        "model_config.d_token": "Token Hidden Dim.",
        "model_config.d_ffn_factor": "Hidden Dim. Factor",
    }
    analysis = PlotAnalysis(
        df,
        save_dir=tmp_path.as_posix(),
        cache=True,
        optim_metrics={"val_acc": "max"},
        numerical_attributes=list(numerical_name_remap),
        categorical_attributes=list(categorical_name_remap),
    )

    attribute_name_remap = {**categorical_name_remap, **numerical_name_remap}
    analysis.make_figures(
        metric_name_remap={
            "val_acc": "Accuracy",
            "val_rmse": "RMSE",
        },
        attribute_name_remap=attribute_name_remap,
    )

    def _exist(p: Path, file_names):
        return (p.joinpath(f"{file_name}.png").exists() for file_name in file_names)

    assert all(
        _exist(tmp_path.joinpath("violinplot", "val_acc"), categorical_name_remap)
    )
    assert all(_exist(tmp_path.joinpath("linearplot", "val_acc"), numerical_name_remap))
    assert not tmp_path.joinpath("linearplot", "val_rmse").exists()
    assert not tmp_path.joinpath("violinplot", "val_rmse").exists()
    no_cache = PlotAnalysis(
        df,
        save_dir=tmp_path.as_posix(),
        cache=False,
        optim_metrics={"val_rmse": Optim.min, "val_acc": "max"},
        numerical_attributes=list(numerical_name_remap.keys()),
        categorical_attributes=list(categorical_name_remap.keys()),
    )
    no_cache.make_figures(
        metric_name_remap={
            "val_acc": "Accuracy",
            "val_rmse": "RMSE",
        },
        attribute_name_remap=attribute_name_remap,
    )

    assert all(
        _exist(tmp_path.joinpath("violinplot", "val_rmse"), categorical_name_remap)
    )
    assert all(
        _exist(tmp_path.joinpath("linearplot", "val_rmse"), numerical_name_remap)
    )
    # checking whether if cache = False, clears memory.
    assert no_cache.cache == None

    with pytest.raises(FileNotFoundError, match="Save directory does not exist"):
        incorrect_path = PlotAnalysis(
            df,
            save_dir=tmp_path.joinpath("nonexistent_folder", "a"),
            cache=False,
            optim_metrics={"val_acc": Optim.min},
            numerical_attributes=list(numerical_name_remap.keys()),
            categorical_attributes=list(categorical_name_remap.keys()),
        )
    with pytest.raises(
        ValueError, match="Must provide a `save_dir` when specifying `cache=True`."
    ):
        no_path = PlotAnalysis(
            df,
            cache=True,
            optim_metrics={"val_acc": Optim.min},
            numerical_attributes=list(numerical_name_remap.keys()),
            categorical_attributes=list(categorical_name_remap.keys()),
        )
    with pytest.raises(
        ValueError,
        match="Must specify a `save_dir` either as an argument to `make_figures` or during class instantiation",
    ):
        PlotAnalysis(
            df,
            optim_metrics={"val_acc": Optim.min},
            numerical_attributes=list(numerical_name_remap.keys()),
            categorical_attributes=list(categorical_name_remap.keys()),
        ).make_figures()
    with pytest.raises(
        ValueError, match="No valid value was found for metric `val_acc`."
    ):
        PlotAnalysis(
            df[df["val_acc"].isna()],
            optim_metrics={"val_acc": Optim.min},
            numerical_attributes=list(numerical_name_remap.keys()),
            categorical_attributes=list(categorical_name_remap.keys()),
        ).make_figures(save_dir=tmp_path.joinpath("file"))
    PlotAnalysis(
        df[df["val_acc"].isna()],
        optim_metrics={"val_rmse": Optim.min},
        numerical_attributes=list(numerical_name_remap.keys()),
        categorical_attributes=list(categorical_name_remap.keys()),
    ).make_figures(save_dir=tmp_path.joinpath("file"))
    assert all(
        _exist(
            tmp_path.joinpath("file", "violinplot", "val_rmse"), categorical_name_remap
        )
    )
    assert all(
        _exist(
            tmp_path.joinpath("file", "linearplot", "val_rmse"), numerical_name_remap
        )
    )
    assert not tmp_path.joinpath("file", "linearplot", "val_acc").exists()
    assert not tmp_path.joinpath("file", "violinplot", "val_acc").exists()
    assert len(
        list(tmp_path.joinpath("file", "violinplot", "val_rmse").glob("*.png"))
    ) == len(categorical_name_remap)
    assert len(
        list(tmp_path.joinpath("file", "linearplot", "val_rmse").glob("*.png"))
    ) == len(numerical_name_remap)


class MockPlot(Plot):
    def _make(self, **kwargs):
        attributes = self.attributes.values
        metric = self.metric.values
        return attributes, metric


def test_plot(tmp_path: Path):
    metric = pd.Series(np.random.randn(100), name="val_acc")
    cat_attributes = pd.DataFrame(
        np.random.randint(100, size=(100, 10)), columns=[np.arange(10)]
    )
    with pytest.raises(ValueError, match="'x' is not a valid Optim"):
        MockPlot(metric=metric, attributes=cat_attributes, metric_obj_fn="x")
    p = MockPlot(metric=metric, attributes=cat_attributes, metric_obj_fn="min")
    _attr, _metric = p._make()
    assert (_attr == cat_attributes).all().all()
    assert (_metric == metric).all()
    metric[:50] = np.nan
    p = MockPlot(metric=metric, attributes=cat_attributes, metric_obj_fn="min")
    _attr, _metric = p._make()
    assert (_attr == cat_attributes[~metric.isna()]).all().all()
    assert (_metric == metric[~metric.isna()]).all()

    fig, ax = p._make_figure()
    assert (ax.get_xticks() == np.linspace(0, 1, len(ax.get_xticklabels()))).all()
    assert ax.get_legend() is None
    p._parse_figure_axis(ax)
    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""
    assert ax.get_legend() is not None
    ax.get_xticklabels()
    p._parse_figure_axis(ax, x_axis="x", y_axis="y", labels=["a", "b", "c"])
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert (ax.get_xticks() == np.arange(3) + 1).all()
    assert (
        np.array([b.get_text() for b in ax.get_xticklabels()])
        == np.array(["a", "b", "c"])
    ).all()


def test_linear_plot():
    metric = pd.Series(np.random.randn(100), name="val_acc")
    num_attributes = pd.DataFrame(np.random.randn(100, 10), columns=[np.arange(10)])
    with pytest.raises(
        ValueError, match="LinearPlot attributes must be single dimensional."
    ):
        p = LinearPlot(metric=metric, attributes=num_attributes, metric_obj_fn="max")
        fig, ax = p._make()

    with pytest.raises(pd.errors.IndexingError, match="Unalignable boolean Series"):
        num_attributes = pd.DataFrame(np.random.randn(110, 1), columns=[np.arange(1)])
        p = LinearPlot(metric=metric, attributes=num_attributes, metric_obj_fn="max")
    num_attributes = pd.DataFrame(np.random.randn(100, 1), columns=[np.arange(1)])
    p = LinearPlot(metric=metric, attributes=num_attributes, metric_obj_fn="max")
    fig, ax = p._make()
    assert len(ax.lines) == 1
    x, y = ax.lines[0].get_data()
    _x = np.array(sorted(num_attributes.values.flatten()))
    _y = np.array(sorted(metric.values.flatten()))
    x = np.array(sorted(x))
    y = np.array(sorted(metric))
    assert np.isclose(y, _y).all()
    # TODO for some reason this is not true, but it is inherit to the seaborn library
    # assert np.isclose(x, _x).all()

    assert max(x) == max(_x) and min(x) == min(_x)

    num_attributes = pd.DataFrame(np.random.randn(100), columns=[np.arange(1)])
    p = LinearPlot(metric=metric, attributes=num_attributes, metric_obj_fn="max")
    fig, ax = p._make()
    assert len(ax.lines) == 1
    fig, ax = p._make()
    assert len(ax.lines) == 2
    fig, ax = p.make()
    assert ax.get_legend() is None


def test_categorical_plot(capture_output, capture_logger):
    out: io.StringIO = capture_logger()
    metric = pd.Series(np.random.randn(100), name="val_acc")
    cat_attributes = pd.Series(np.random.randint(10, size=(100)), name="attr")
    attribute_metric_map = Categorical._make_attribute_metric_map(
        metric, cat_attributes
    )

    assert set(attribute_metric_map.keys()) == set(range(10))

    assert np.isclose(
        sorted(pd.concat(attribute_metric_map.values())), sorted(metric)
    ).all()
    _, counts = np.unique(cat_attributes, return_counts=True)
    _counts = np.array([len(attribute_metric_map[i]) for i in range(10)])
    assert (counts == _counts).all()

    # sort_vals tests
    metric_min = Categorical._sort_vals_obj(None, metric, "min")
    metric_max = Categorical._sort_vals_obj(None, metric, "max")
    assert np.isclose(metric_min, np.sort(metric)).all()
    assert np.isclose(metric_max, np.sort(metric)[::-1]).all()
    metric[:10] = np.nan
    metric_min = Categorical._sort_vals_obj(None, metric, "min")
    metric_max = Categorical._sort_vals_obj(None, metric, "max")
    assert np.isnan(metric_min[-10:]).all()
    assert np.isnan(metric_max[-10:]).all()

    # missing values ignored
    cat_attributes = cat_attributes.astype(object)
    cat_attributes.iloc[:10] = None
    cat_attributes.iloc[10:20] = np.nan
    cat_attributes.iloc[20:30] = "None"
    cat_attributes.iloc[30:40] = "Type(None)"
    cat_attributes.iloc[40:50] = "nan"
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Type(None), and `None` are both present as categorical values. Unable to rename None value."
        ),
    ):
        attribute_metric_map = Categorical._make_attribute_metric_map(
            metric, cat_attributes
        )

    cat_attributes.iloc[30:40] = "Type(None)s"

    metric = pd.Series(np.random.randn(100), name="val_acc")

    stdout, stderr = capture_output(
        lambda: Categorical._make_attribute_metric_map(metric, cat_attributes)
    )
    assert (
        "`None` is present as a categorical string value as well as None. Will rename None to Type(None)."
        in out.getvalue()
    )

    attribute_metric_map = Categorical._make_attribute_metric_map(
        metric, cat_attributes
    )

    for i, v in enumerate(["Type(None)", np.nan, "None", "Type(None)s", "nan"]):
        assert np.isclose(
            attribute_metric_map[v].values, metric[i * 10 : (i + 1) * 10].values
        ).all()


def test_violin_plot():
    metric = pd.Series(np.random.randn(100), name="val_acc")
    cat_attributes = pd.DataFrame(np.random.randint(10, size=(100, 10)))
    with pytest.raises(
        ValueError, match="ViolinPlot attributes must be single dimensional."
    ):
        p = ViolinPlot(metric=metric, attributes=cat_attributes, metric_obj_fn="max")
        fig, ax = p._make()
    with pytest.raises(pd.errors.IndexingError, match="Unalignable boolean Series"):
        cat_attributes = pd.Series(np.random.randn(110), name="attr")
        p = ViolinPlot(metric=metric, attributes=cat_attributes, metric_obj_fn="max")

    n_categories = 10
    cat_attributes = pd.Series(np.random.randint(n_categories, size=(100)), name="attr")
    p = ViolinPlot(metric=metric, attributes=cat_attributes, metric_obj_fn="max")
    fig, ax = p._make()
    assert len(ax.lines) == n_categories * 2
    x, (y_min, y_max) = ax.lines[0].get_data()
    assert y_min == min(p.attribute_metric_map[0])
    assert y_max == max(p.attribute_metric_map[0])
    assert all(
        l.get_text()
        == f"Mean: {p.attribute_metric_map[i].mean():.2e}\nBest: {p.attribute_metric_map[i].max():.2e}\n{i}"
        for i, l in enumerate(ax.get_xticklabels())
    )


def test_get_best_results_by_metric():
    # Create a sample dataframe
    raw_results = pd.DataFrame({
        "trial_uid": [1, 1, 2, 2, 3],
        "metric_1": [0.5, 0.6, 0.7, 0.8, 0.9],
        "metric_2": [0.3, 0.4, 0.2, 0.1, 0.5]
    })

    metric_map = {
        "metric_1": Optim.min,
        "metric_2": Optim.max
    }

    # Call the _get_best_results_by_metric method
    result_df = Analysis._get_best_results_by_metric(raw_results, metric_map)

    # Define expected results
    expected_columns = ["trial_uid", "metric_1", "metric_2", "best"]

    # Check if the result DataFrame has the expected columns
    assert list(result_df.columns) == expected_columns

    # Check if the best results are selected correctly for each metric
    assert result_df.loc[result_df["best"] == "metric_1", "metric_1"].min() == 0.5
    assert result_df.loc[result_df["best"] == "metric_2", "metric_2"].max() == 0.5

def test_remap_results():
    # Create sample dataframes
    attributes = pd.DataFrame({"color": ["red", "blue"], "size": [10, 20]})
    metrics = pd.DataFrame({"loss": [0.5, 0.4], "accuracy": [0.8, 0.9]})
    metric_map = {"loss": Optim.min, "accuracy": Optim.max}
    metric_name_remap = {"loss": "error", "accuracy": "acc"}
    attribute_name_remap = {"color": "c", "size": "s"}

    # Call the _remap_results method
    remapped_attrs, remapped_metrics, updated_map = Analysis._remap_results(
        attributes, metrics, metric_map,
        metric_name_remap=metric_name_remap,
        attribute_name_remap=attribute_name_remap
    )

    # Check if the remapping worked correctly
    assert list(remapped_attrs.columns) == ["c", "s"]
    assert list(remapped_metrics.columns) == ["error", "acc"]
    assert updated_map == {"error": Optim.min, "acc": Optim.max}

def test_analysis_numerical_attributes_none():
    # Create a sample DataFrame (for testing purposes)
    sample_data = {'color': ['red', 'blue'], 'size': [10, 20]}
    sample_df = pd.DataFrame(sample_data)

    # Define the categorical attributes
    categorical_attributes = ['color']

    # Define the optimization metrics
    optim_metrics = {'loss': 'min', 'accuracy': 'max'}

    # This test expects a ValueError to be raised when numerical_attributes is None
    with pytest.raises(ValueError, match="Must provide `_numerical_attributes`"):
        Analysis(results=sample_df, categorical_attributes=categorical_attributes, numerical_attributes=None, optim_metrics=optim_metrics)

def test_analysis_categorical_attributes_none():
    # Create a sample DataFrame (for testing purposes)
    sample_data = {'color': ['red', 'blue'], 'size': [10, 20]}
    sample_df = pd.DataFrame(sample_data)

    # Define the numerical attributes
    numerical_attributes = ['size']

    # Define the optimization metrics
    optim_metrics = {'loss': 'min', 'accuracy': 'max'}

    # This test expects a ValueError to be raised when categorical_attributes is None
    with pytest.raises(ValueError, match="Must provide `categorical_attributes`"):
        Analysis(results=sample_df, categorical_attributes=None, numerical_attributes=numerical_attributes, optim_metrics=optim_metrics)

def test_analysis_optim_metrics_none():
    # Create a sample DataFrame (for testing purposes)
    sample_data = {'color': ['red', 'blue'], 'size': [10, 20]}
    sample_df = pd.DataFrame(sample_data)

    # Define the categorical attributes
    categorical_attributes = ['color']

    # Define the numerical attributes
    numerical_attributes = ['size']

    # This test expects a ValueError to be raised when optim_metrics is None
    with pytest.raises(ValueError, match="Missing `optim_metrics` or unable to derive from supplied results."):
        Analysis(results=sample_df, categorical_attributes=categorical_attributes, numerical_attributes=numerical_attributes, optim_metrics=None)

def test_parse_results_results(tmp_path: Path,ablator_results):
    # Create a sample Results object
    df = pd.DataFrame(ablator_results)
    results = Results(config=None, experiment_dir="", cache=False, use_ray=False)
    results.data = df
    results.categorical_attributes = ["experiment_name"]
    results.numerical_attributes = ["batch_size"]
    results.metric_map = {"accuracy": Optim.max, "loss": Optim.min}

    # Call the _parse_results method
    parsed_df, cat_attrs, num_attrs, metrics = Results._parse_results(results)

    # Define expected results
    expected_df = df
    expected_cat_attrs = ["experiment_name"]
    expected_num_attrs = ["batch_size"]
    expected_metrics = {"accuracy": Optim.max, "loss": Optim.min}

    # Check if the returned values match the expected ones
    assert parsed_df.equals(expected_df)
    assert cat_attrs == expected_cat_attrs
    assert num_attrs == expected_num_attrs
    assert metrics == expected_metrics

# def test_results_init1(tmp_path,make_config):
#     # Create a temporary experiment directory
#     experiment_dir = tmp_path / "tmp_experiment"
#     experiment_dir.mkdir()

#     # # Create a mock config file
#     # with open(experiment_dir / "default_config.yaml", "w") as f:
#     #     f.write("some_config: value")

#     # # Create a test configuration
#     # class TestConfig(ParallelConfig):  # Using ConfigBase for simplicity
#     #     search_space = {
#     #         "param1": SearchSpace(value_range=[0, 1], value_type="float"),
#     #         "param2": SearchSpace(value_range=[0, 1], value_type="float"),
#     #     }
#     #     optim_metrics = {"metric1": Optim.max}

#     # Initialize Results class
#     results = Results(config=make_config, experiment_dir=experiment_dir)

#     # Add assertions as needed
#     assert isinstance(results.config, make_config)
#     assert isinstance(results.experiment_dir, Path)
#     assert isinstance(results.metric_map, dict)
#     assert isinstance(results.data, pd.DataFrame)
#     assert isinstance(results.config_attrs, list)
#     assert isinstance(results.search_space, dict)
#     assert isinstance(results.numerical_attributes, list)
#     assert isinstance(results.categorical_attributes, list)

# def test_results_init(tmp_path,make_config):
#     # Create a temporary experiment directory
    
#     experiment_dir = tmp_path / "tmp_experiment"
#     experiment_dir.mkdir()

#     # Create a ParallelConfig instance
   

#     # Initialize Results class
#     results = Results(config=config, experiment_dir=experiment_dir)

#     # Add assertions as needed
#     assert isinstance(results.config, ParallelConfig)
#     assert isinstance(results.experiment_dir, Path)
#     assert isinstance(results.metric_map, dict)
#     assert isinstance(results.data, pd.DataFrame)
#     assert isinstance(results.config_attrs, list)
#     assert isinstance(results.search_space, dict)
#     assert isinstance(results.numerical_attributes, list)
#     assert isinstance(results.categorical_attributes, list)

# def test_init_results(ablator_results):
#         results: Results = ablator_results
#         assert isinstance(results.metric_map, dict)
#         assert isinstance(results.data, pd.DataFrame)
#         assert isinstance(results.config_attrs, list)
#         assert isinstance(results.search_space, dict)
#         assert isinstance(results.numerical_attributes, list)
#         assert isinstance(results.categorical_attributes, list)


def test_results_config_not_parallel():
    # Create a test configuration that is not a subclass of ParallelConfig
    class TestConfig(ConfigBase):
        pass

    # Create a temporary experiment directory
    experiment_dir = "./tmp_experiment"
    os.makedirs(experiment_dir, exist_ok=True)
    try:
        results = Results(config=TestConfig, experiment_dir=experiment_dir)
    except ValueError as e:
        assert str(e) == (
            "Configuration must be subclassed from ``ParallelConfig``. "
        )
    else:
        raise AssertionError("ValueError was not raised")
    
def test_results_single_trial_config():
    # Create a test configuration that is a single-trial config
    class TestConfig(RunConfig):
        pass

    # Create a temporary experiment directory
    experiment_dir = "./tmp_experiment"
    os.makedirs(experiment_dir, exist_ok=True)

    # Check if ValueError is raised
    try:
        results = Results(config=TestConfig, experiment_dir=experiment_dir)
    except ValueError as e:
        assert str(e) == (
            "Provided a ``RunConfig`` used for a single-trial. Analysis "
            "is not meaningful for a single trial. Please provide a ``ParallelConfig``."
        )
    else:
        raise AssertionError("ValueError was not raised")


def test_metric_names(ablator_results):
    #Initialize a Results object 
    results:Results= ablator_results

    # Set metric_map with 
    results.metric_map = {"metric1": "max", "metric2": "min"}

    # Call metric_names method
    metrics = results.metric_names

    # Check if the returned list is correct
    assert metrics == ["max", "min"]

def test_read_result_no_results_found():
    # Assuming experiment_dir is a non-existent directory
    experiment_dir = Path("nonexistent_directory")
    config_type = None  

    # Use pytest.raises to check if the RuntimeError is raised
    with pytest.raises(RuntimeError) as exc_info:
        Results.read_results(config_type, experiment_dir)

    # Check the error message
    assert str(exc_info.value) == f"No results found in {experiment_dir}"

# def test_assert_cat_attributes(make_config,sample_example_dir):
#     custom_config = make_config(tmp_path="/tmp/experiments/")

#     # Initialize a Results object
#     results = Results(custom_config, experiment_dir="")

#     # Create a DataFrame for testing
#     test_data = {
#         "attr1": ["A", "A", "B", "C", "C", "C"],
#         "attr2": ["X", "X", "Y", "Y", "Z", "Z"]
#     }
#     results.data = pd.DataFrame(test_data)

#     # Call _assert_cat_attributes method with a list of categorical attributes
#     categorical_attributes = ["attr1", "attr2"]
#     results._assert_cat_attributes(categorical_attributes)

#     # The above method should not raise any AssertionError

#     # Create an imbalanced dataset for testing
#     test_data_imbalanced = {
#         "attr1": ["X", "X", "Y", "Y", "Z", "Z"],
#         "attr2": ["X", "X", "Y", "Y", "Z", "Z"]
#     }
#     results.data = pd.DataFrame(test_data_imbalanced)

#     # Call _assert_cat_attributes method with a list of categorical attributes
#     with pytest.warns(UserWarning):
#         results._assert_cat_attributes(categorical_attributes)


def test_image_uniqueness():

    # Create first plot
    metric = pd.Series(np.random.randn(100), name="val_acc")
    num_attributes = pd.DataFrame(np.random.randn(100, 1), columns=[np.arange(1)])
    p1 = LinearPlot(metric=metric, attributes=num_attributes, metric_obj_fn="max")
    fig1, ax1 = p1._make()

    # Create second plot with different data
    metric2 = pd.Series(np.random.randn(100), name="val_acc")
    num_attributes2 = pd.DataFrame(np.random.randn(100, 1), columns=[np.arange(1)])
    p2 = LinearPlot(metric=metric2, attributes=num_attributes2, metric_obj_fn="max")
    fig2, ax2 = p2._make()

    # Save plots
    results_directory = "results_directory"
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    plot1_path = f"{results_directory}/plot1.png"
    plot2_path = f"{results_directory}/plot2.png"
    fig1.savefig(plot1_path)
    fig2.savefig(plot2_path)

    # Ensure plots are properly closed after saving
    plt.close(fig1)
    plt.close(fig2)

    # Check uniqueness of the two images by comparing file contents
    with open(plot1_path, 'rb') as f1, open(plot2_path, 'rb') as f2:
        assert f1.read() != f2.read(), "Images are not unique"



if __name__ == "__main__":
    import copy
    from tests.conftest import run_tests_local

    l = locals()
    fn_names = [fn for fn in l if fn.startswith("test_")]
    test_fns = [l[fn] for fn in fn_names]
    run_tests_local(test_fns)
   


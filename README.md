<img width="340" alt="Ablator Image" src="https://github.com/apneetha/ablator/assets/111663232/f6b7d5b6-6c52-403d-b7bc-2e6156e64b2b"> 

[![Documentation](https://img.shields.io/badge/docs-Ablator_website-blue)](https://dashboard.ablator.online/landing)
[![Version](https://img.shields.io/badge/version-1.0.1-blue)](https://github.com/fostiropoulos/ablator)
[![Downloads](https://img.shields.io/github/downloads/user/repo/total)](LINK_TO_RELEASES)
[![Twitter Follow](https://img.shields.io/twitter/follow/username?label=Follow&style=social)](https://twitter.com/ABLATOR_ORG)
[![Discord](https://img.shields.io/discord/YOUR_SERVER_ID?label=Discord&logo=discord&color=7289DA)](https://discord.com/invite/9dqThvGnUW)
[![Slack](https://img.shields.io/badge/Slack-Join%20Us-blue?logo=slack)](https://ablator.slack.com/join/shared_invite/zt-23ak9ispz-HObgZSEZhyNcTTSGM_EERw#/shared-invite/email)


ABLATOR is a tool designed to assist in the horizontal scaling of experimental trials for machine learning models, automating the process of configuring and conducting multiple experiments with various hyperparameter settings. The tool streamlines model experimentation, offers a flexible configuration system, and aids in result interpretation through visualization.


Ablation studies involve removing specific parts of a neural network architecture or changing different aspects of the training process to examine their contributions to the model's performance.

[ReadMe Figure]

Learn more about [Ablator Module]():

- [Configuration Module](https://docs.ablator.online/notebooks/configuration-basics#Configuration-categories): Text about the module
- [Training Module]():Text about the module
- [Experiment Result metrics module](): Text about the module
- [Analysis Module](https://docs.ablator.online/notebooks/interpreting-results): Text about the module


### What does Ablator Offer?

Comparison table with existing framework:

| Framework      | HPO            | Configuration  | Training       | Tuning         | Analysis       |
|----------------|----------------|----------------|----------------|----------------|----------------|
| Ray            | :white_check_mark:     | :x:         | :x:         | :white_check_mark:     | :x:         |
| Lighting       | :x:         | :x:         | :white_check_mark:     | :x:         | :x:         |
| Optuna         | :white_check_mark:     | :x:         | :x:         | :x:         | :white_check_mark:     |
| Hydra          | :x:         | :white_check_mark:     | :x:         | :x:         | :x:         |
| **ABLATOR** | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

Features compared, hyperparameter selection (`HPO`), removing boilerplate code for configuring experiments (`Configuration`), removing boiler plate code for running experiments at scale (`Tuning`) and performing analysis on the hyperparameter selection (`Analysis`).

Using:
1. Ray: You will need to write boiler-plate code for integrating with a configuration system (i.e. Hydra), saving experiments artifacts or logging (i.e. integrate with Wandb).

2. Lighting: You will need to write boiler-plate code for HPO (i.e. using Optuna), Configuring experiments (i.e. Hydra) and horizontal distributed execution (i.e. integrate with Ray)

3. Hydra: The configuration system is not strongly typed (ABLATOR), and does not provide support for common ML use-cases where configuration attributes are **Derived** (inferred during run-time) or **Stateless** (change between trials). Additionally, ABLATOR provides support for custom objects that are dynamically inferred and initialized during execution.

4. ABLATOR: Combines Ray back-end, with Optuna for HPO and removes boiler-plate code for fault tollerant strategies, training, and analyzing the results.

Integrating different tools, for distributed execution, fault tollerance, training, checkpointing and analysis is **error prone**! Poor compatibility between tools, verisioning errors will lead to errors in your analysis.


You can use ABLATOR with any other library i.e. PyTorch Lighting. Just wrap a Lighting model with ModelWrapper. For examples please look [examples](examples)


Spend more time in the creative process of ML research and less time on dev-ops.

### How to get started with Ablator?

### Install

Use a python virtual enviroment to avoid version conflicts.

`pip install git+https://github.com/fostiropoulos/ablator.git`

For Development

1. `git clone git@github.com:fostiropoulos/ablator.git`
2. `cd ablator`
3. `pip install -e .[dev]`


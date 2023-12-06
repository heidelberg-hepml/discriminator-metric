# Discriminator Metric for Generative Models

This tool makes it easy to train a classifier to reweight samples from a generative
model. It comes with an extensive plotting pipeline that allows to inspect the classifier
output to evaluate the performance of the generative network.

This is the reference repository for the article _"How to Understand Limitations of Generative Networks"_.
The preprint is available at [https://arxiv.org/abs/2305.16774](https://arxiv.org/abs/2305.16774)

We collect here all the datasets used to train the classifiers.

Samples obtained from the generative models and truth generated events:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10277550.svg)](https://doi.org/10.5281/zenodo.10277550)

Original JetNet dataset:
Kansal, R., Duarte, J., Su, H., Orzari, B., Tomei, T., Pierini, M., Touranakou, M., Vlimant, J.-R., & Gunopulos, D. (2022). JetNet (Versione 2). Zenodo. [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6975118.svg)](https://doi.org/10.5281/zenodo.6975118)

CaloGAN dataset:
Krause, C., & Shih, D. (2021). Electromagnetic Calorimeter Shower Images of CaloFlow. Zenodo. [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5904188.svg)](https://doi.org/10.5281/zenodo.5904188)

## Usage

Training a discriminator:
```
python -m src params/prec_inn.yaml
```

Loading a trained model:
```
python -m src --load_model 20230303_100000_run_name
```

Loading a specific trained model (e.g. best model, final model, checkpoint):
```
python -m src --load_model --model_name=final 20230303_100000_run_name
```

Loading a trained model, but compute weights again:
```
python -m src --load_model --load_weights 20230303_100000_run_name
```

## Parameters

The following parameters can be set in the YAML parameter file.

### Data loader

Parameter       | Description
----------------|-------------------------------------------------
`loader_module` | Name of the data loader module in src.loader
`loader_params` | Data-loader specific parameters

### Architecture

Parameter        | Description
-----------------|----------------------------------------------------------------------------
`layers`         | Number of layers
`hidden_size`    | Number of nodes of the hidden layers
`activation`     | Activation function. `relu`, `leaky_relu` or `elu`
`dropout`        | Dropout fraction
`prior_prec`     | Gaussian prior standard deviation of the Bayesian network
`std_init`       | Logarithm of the initial standard deviation of the Bayesian network weights
`negative_slope` | Negative slope of the leaky ReLU activation

### Training

Parameter         | Description
------------------|---------------------------------------------------------------------------
`bayesian`        | Train as a Bayesian network
`batch_size`      | Batch size
`lr`              | Initial learning rate
`betas`           | Adam optimizer betas
`eps`             | Adam optimizer eps
`weight_decay`    | L2 weight decay
`lr_scheduler`    | Type of LR scheduler: `one_cycle`, `step`, `reduce_on_plateau`
`max_lr`          | One Cycle scheduler: maximum LR
`lr_decay_epochs` | Step scheduler: Epochs after which to reduce the LR
`lr_decay_factor` | Step and reduce on plateau schedulers: Decay factor
`lr_patience`     | Reduce on plateau scheduler: Number of epochs without improvement for reduction.
`epochs`          | Number of epochs
`train_samples`   | Total number of samples used for training (alternative to number of epochs)
`checkpoint_interval` | If value n set, save the model after every n epochs

### Evaluation

Parameter          | Description
-------------------|-------------------------------------------------------------
`bayesian_samples` | Number of samples to draw from the network weight posterior
`lower_thresholds` | List of lower weight thresholds for the clustering plots
`upper_thresholds` | List of upper weight thresholds for the clustering plots

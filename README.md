## Usage

Training a discriminator:
```
python -m src params/prec_inn.yaml
```

Loading a trained model:
```
python -m src --load_model 20230303_100000_run_name
```

Loading a trained model, but compute weights again:
```
python -m src --load_model --load_weights 20230303_100000_run_name
```

## Parameters

### Data loader

Parameter      | Description
---------------|-------------------------------------------------
loader\_module | Name of the data loader module in src.loader
loader\_params | Data-loader specific parameters

### Architecture

Parameter       | Description
----------------|-----------------------------------------------------------------------------
layers          | Number of layers
hidden\_size    | Number of nodes of the hidden layers
activation      | Activation function. `relu`, `leaky_relu` or `elu`
dropout         | Dropout fraction
prior\_prec     | Gaussian prior standard deviation of the Bayesian network
std\_init       | Logarithm of the initial standard deviation of the Bayesian network weights
negative\_slope | Negative slope of the leaky ReLU activation

### Training

Parameter         | Description
------------------|--------------------------------------------------------
bayesian          | Train as a Bayesian network
batch\_size       | Batch size
lr                | Initial learning rate
betas             | Adam optimizer betas
eps               | Adam optimizer eps
weight\_decay     | L2 weight decay
lr\_scheduler     | Type of LR scheduler: `one_cycle` or `step`
max\_lr           | One Cycle scheduler: maximum LR
lr\_decay\_epochs | Step scheduler: Epochs after which to reduce the LR
lr\_decay\_factor | Step scheduler: Decay factor
epochs            | Number of epochs

### Evaluation

Parameter         | Description
------------------|-------------------------------------------------------------
bayesian\_samples | Number of samples to draw from the network weight posterior
lower\_thresholds | List of lower weight thresholds for the clustering plots
upper\_thresholds | List of upper weight thresholds for the clustering plots

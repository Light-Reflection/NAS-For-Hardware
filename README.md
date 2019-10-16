# Quick Start

This section walks you through a simple use case of Model Generator on CIFAR-10 dataset. You should be in a good starting point of further exploring the richer features in `ModelGenerator` after this tutorial.

## Preparation

To start with, you can use git to clone our project from [gitlab](https://gitlab.dm-ai.cn/research-hardware/edge-deployment/dmmo):

```
git clone https://gitlab.dm-ai.cn/research-hardware/edge-deployment/dmmo.git
pip install -e dmmo
```





## Model Generator Pipeline Explanation

The work flow with the code execution is illustrated as below:

<img src="/home/dm/Desktop/dmmodoc/doc/source/imgs/workflow.jpg" alt="Workflow" style="zoom: 50%;" />

### Initialize SuperNet

A `runner` is bounded with a specified DNN model and is initialized with dataset name and training hyper-parameters. It then loads a model checkpoint and links required GPU resources, as demonstrated below.

```
# initialize a SuperNet
import dmmo as dm
model = dm.supernet(num_of_classes = 100,name = 'testing').cuda()
```

### Train & Search 

Train and Search is defined down below. You can change parameters in fit function to adjust training and searching stage.

```
# define traning parameters and search parameters
model.fit(model= model,
         data_root = '/home/dm/data/CIFAR',
         epoch = 100)
```

### Save thinned model and evaluate

In the end, the best model structure will be full-trained and saved in 'save_path', the performance can be evaluated by evaluate function.

```
# save trained model
model.save(save_path = 'save_path')
model.evaluate
```

### Put Everything Together

Finally, a `main` function wraps up the pieces and bits we mentioned above and deliver a big picture for the entire procedure.

```
# initialize a SuperNet
import dmmo as dm
model = dm.supernet(num_of_classes = 100,name = 'testing').cuda()

# define traning parameters and search parameters
model.fit(model= model,
         data_root = '/home/dm/data/CIFAR',
         epoch = 100)
         
# save trained model

model.save(save_path = 'save_path')
model.evaluate
```

https://gitlab.dm-ai.cn/yuanliuchun/DMMO-MG


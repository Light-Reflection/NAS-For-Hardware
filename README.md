# Model-Generator

[Modelgenerator](https://gitlab.dm-ai.cn/yuanliuchun/DMMO-MG) is a Python package for neural network pruning application.

Deep learning models are widely deployed in various applications, such as image recognition, machine translation, and speech synthesis. However, current models pool can't satisfy personal taste,  large amount of efforts have spent on designing suitable neural networks in different situation, thus we  provides an easy-to-use toolkit to personalize models with high accuracy, smaller latency and less parameters.

## Framework

![Pipeline](/home/dm/Desktop/dmmodoc/doc/source/imgs/pipline.jpg)

`Generator`  and`Evaluator` are key building blocks in `ModelGenerator` , `Generator`will firstly produce a supernet then it will be trained and evaluated by `Evalueator`.

| Evaluator           | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| Action Space-random |                                                              |
| Action Space-random | Searching best pruning rate for each layer using greedy algorithm([Abbasi-Asl et al., 2017](https://arxiv.org/abs/1705.07356)) |
| Action Space-PTE    |                                                              |
| RL                  | Efficient Neural Architecture Search via Parameter Sharing([Hieu Pham., 2019](https://arxiv.org/pdf/1802.03268.pdf)) |

## Highlighted Features

If your backbone DNN models consist of nothing else but operators listed below. `ModelBarber` may perfectly support your models with or without some tiny twists:

### Operators

- Convolution
- Fully-Connected
- Short-Cut Connection
- Depthwise Separable Convolution

Coming soon: LSTM and Transformer

`ModelBarber` natively supports below applications as seen in the [example practice](http://192.168.7.7/dmmo/md/ModelBarber.md.html#one-single-example) section. However, as before-mentioned, any other application that are not listed below may be perfectly supported if their backbone models are built with supported operators listed above.

### Applications

- Image Classification
- Object Detection



## Competition with competitors－based on cifar-10

Here is a comparison between `ModelBarber` and alternatives from other companies. The table below shows the comparison of running platform and supported feature.

| Experiment       | Accuracy | Parameters | Latency | Resource | Ｍethod |
| ---------------- | -------- | ---------- | :------ | -------- | ------- |
| Google AutoML    |          |            |         |          |         |
| Baidu AutoML     |          |            |         |          |         |
| Autokeras        |          |            |         |          |         |
| NNI              |          |            |         |          |         |
| HuaWei ModelArts |          |            |         |          |         |
| DMMO             |          |            |         |          |         |


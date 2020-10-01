# caloGraphNN

Repository that contains minimal implementations of the graph neural network layers discussed in [arxiv:1902.07987].
The code provided here is using tensorflow and or keras.
For a pytorch implementation, please refer to: https://github.com/rusty1s/pytorch_geometric

For tensorflow and keras, all necessary functions are included in the individual python files in this repository. No further dependencies are needed. The layers can be used analogously to tensorflow layers.
The bare layers can be found in caloGraphNN.py, and can be used in a similar way as bare tensorflow layers, and therefore can be easily implemented in custom DNN architectures.
The source code for models described in the paper is in tensorflow_models.py for reference.

The keras implementation of the layers and models can be found in the files: caloGraphNN_keras.py, keras_models.py.

Both implementations require at least tensorflow 1.8.

When using these layers to build models or modifying them, please cite our paper:

```
@article{Qasim:2019otl,
      author         = "Qasim, Shah Rukh and Kieseler, Jan and Iiyama, Yutaro and
                        Pierini, Maurizio",
      title          = "{Learning representations of irregular particle-detector
                        geometry with distance-weighted graph networks}",
      journal        = "Eur. Phys. J.",
      volume         = "C79",
      year           = "2019",
      number         = "7",
      pages          = "608",
      doi            = "10.1140/epjc/s10052-019-7113-9",
      eprint         = "1902.07987",
      archivePrefix  = "arXiv",
      primaryClass   = "physics.data-an",
      SLACcitation   = "%%CITATION = ARXIV:1902.07987;%%"
}
```

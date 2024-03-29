# AI Project 4: Learning

## Cooperators:
Yu Zhang, Jiupeng Zhang, Lu Zhang


## Dependencies:
torch, numpy, enum, collections


## How to run:

### Decision tree
there are two modes: 1 is the training mode, 2 is the testing mode.
you can just copy the command to terminal to test our code
```bash
    $ python3 decision_tree_generator.py data/AIMA_Restaurant-desc.txt 1
    $ python3 decision_tree_generator.py data/AIMA_Restaurant-desc.txt 2
    $ python3 decision_tree_generator.py data/tic-tac-toe.data.txt 1
    $ python3 decision_tree_generator.py data/tic-tac-toe.data.txt 2
    $ python3 decision_tree_generator.py data/iris.data.discrete.txt 1
    $ python3 decision_tree_generator.py data/iris.data.discrete.txt 2
```


## Neural network
there are also two modes: 1 is the training mode, 2 is the testing mode.
you can just copy the command to terminal to test our code
```bash
    $ python3 pt_mlp_classifier.py data/iris.data.txt 1
    $ python3 pt_mlp_classifier.py data/iris.data.txt 2
    $ python3 pt_mlp_classifier.py data/tic-tac-toe.data.txt 1
    $ python3 pt_mlp_classifier.py data/tic-tac-toe.data.txt 2
```
**Notice: you must train the model before test it!**
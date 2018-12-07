# AI Project 4: Learning

In this project you will have the opportunity to implement and evaluate one or more machine learning algorithms. This is a huge area of application and research, and is well-covered in upper-level Computer Science courses. It’s also nearing the end of the term, so we can really only scratch the surface with this project. But remember, the more you do and the more you do yourself, the more you’ll learn.

At minimum, I would like you to implement the decision tree learning algorithm from the textbook (AIMA 18.3) and perform some light experiments.

For additional credit, you may experiment with your own implementation of a multilayer perception using the same data. Your implementation may be built from scratch, using my code as a base, or even by relying on Tensorflow – as long as you can explain how it works.

You should test your implementation(s) on the Iris dataset (https://archive.ics.uci.edu/ml/datasets/iris) and another dataset of your choice. An important part of your grade for this assignment will be the quality of the writeup describing your experiments. You need to convince us that your implementation works AND that you know how to evaluate a machine learning program. We should be able to read your writeup, then run the program(s) and compare the output with the results in your writeup, then look at your code to see how you did it. Please craft your writeup and your programs so that this is easy for us. You want the TAs to be happy.


## Decision Tree Learning
Decision tree learning is well covered in AIMA Sect. 18.3.

You should think about what it takes to represent a decision tree in a computer program. Think about it now.

Ok, we hope you thought about trees, whose nodes are either attributes to test (at the internal nodes) or values to return (at the leaves), as well as the attributes and their domains of values themselves. You can easily write classes (or structures) to represent these things.

Implement the entropy-based attribute selection method described in Sect. 18.3.4. (You might do something simpler first, but for full points you must do the real calculation, which is the basis of the ID3 algorithm.) You may restrict your datasets to only those involving categorical (i.e., discrete) attributes.


## Neural Networks
Neural networks are covered in AIMA Section 18.7. It does cover all the important definitions for both single-layer and multi-layer feed-forward networks, and it provides the algorithm for backpropagation in multi-layer networks (Fig. 18.24). That said, it is very concise. So if you choose to implement this type of learner, be prepared to do some thinking and/or additional research as you develop and evaluate your system.

Think about what you need to represent a neural network in software.

Ok. We hope you thought about “units,” layers, connections, weights, activation functions, inputs, and outputs. It is not hard to design classes incorporating these elements. However I suggest that you understand how the backpropagation algorithm works be- fore you lock in your design. In particular, note that it requires that you be able to go both forward and backward through the layers of your networks, even though the network is “feed-forward.”

As the textbook says: “Unfortunately, for any particular network structure, it is harder to characterize exactly which functions can be represented and which ones cannot.” In other words, designing neural networks is something of an art. Whatever you do, document it clearly (what and why) in your writeup and include graphs as necessary to support your description. Should you choose to pursue this extra credit option, your documentation should describe the number of hidden layers, the number of units per hidden layer, the activation function, the loss function, training method (and hyper-parameters). A plot of accuracy over time (or iterations) would be an excellent addition.

As always, you will learn the most if you develop your implementation yourself; however, you may use my code as a starting point. Note that my code was designed for regression, not classification, so you will need to modify it at least somewhat. There may be bugs – they are not intentional, and if you let me know I will try to fix them, but you are responsible for being able to identify when something is “working” and when it is not. A final warning: MLPs for classification perform best with softmax and cross-entropy. You can use cross-entropy with my code verbatim, but you may need to modify backprop to incorporate softmax!

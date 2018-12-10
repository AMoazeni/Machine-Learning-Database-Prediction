# Artificial Neural Networks


Before we look at the Database Prediction problem and start programming, let's take a look at the theory behind the Artificial Neural Network algorithm which was popularized by Geoffrey Hinton in the 1980's and is used in Deep Machine Learning. "Deep" in Deep Learning refers to all the hidden layers used in this type of Dynamic Programming algorithm.



The input layer observations and related output refer to ONE row of data. Adjustment of weights is how Neural Nets learn, they decide the strength and importance of signals that are passed along or blocked by an Activation Function. They keep adjusting weights until the predicted output closely matches the actual output.



<img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Database-Prediction/master/Images/01%20-%20Deep%20Learning.png">



Here is a zoomed in version of the node diagram. Yellow nodes represent inputs, green nodes are the hidden layers, and red nodes are outputs.



<img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Database-Prediction/master/Images/02%20-%20Neuron.png">



Feature Scaling (Standardize or Normalize) is applied to input variables makes it easy for Neural Nets to process data by bringing their values close to each other, read 'Efficient Back Propagation.pdf' in the research papers section.



\begin{equation*}
X_{Standardized} = \frac{X - Min(X)}{Max(X) - Min(X)} 
\end{equation*}



$$
X_{Normalized} = \frac{X - μ (Mean)}{σ (Standard Deviation)} 
$$





# Activation Function


Here is a list of some Neural Network Activation Functions. Read 'Deep sparse rectifier neural networks.pdf' in the research papers section.



1. Threshold Function - Rigid binary style function
<img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Database-Prediction/master/Images/03%20-%20Threshold.png" width="400">

2. Sigmoid Function - Smooth, good for output Layers that predict probability
<img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Database-Prediction/master/Images/04%20-%20Sigmoid.png" width="400">

3. Rectifier Function - Gradually increases as input Value increases
<img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Database-Prediction/master/Images/05%20-%20Rectifier.png" width="400">

4. Hyperbolic Tangent Function - Similar to Sigmoid Function but values can go below zero
<img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Database-Prediction/master/Images/06%20-%20Tanh.png" width="400">



Different layers of a Neural Net can use different Activation Functions.



<img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Database-Prediction/master/Images/07%20-%20NN%20Activation%20Example.png" width="600">





# Cost Function


The Cost Function is a plot of the differences between the target and the network's output, which we try to minimize through weight adjustments (Backpropagation) in epochs (one training cycle on the Training Set). Once input information is fed through the network and a y_hat output estimate is found (Forward-propagation), we take the error and go back through the network and adjust the weights (Backpropagation Algorithm). The most common cost function is the Quadratic (Root Mean Square) cost:



$$
Cost = \frac{(\hat y - y)^2}{2} = \frac{(Wighted Estimate - Actual)^2}{2} 
$$



Read this [Deep Learning Book](http://neuralnetworksanddeeplearning.com/index.html) and this [List of Cost Functions Uses](https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications?).





# Batch Gradient Descent


This is a Cost minimization technique that looks for downhill slopes and works on Convex Cost Functions. The function can have any number of dimensions, but we are only able to visualize up to three dimensions.


### 1D Gradient Descent
<img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Database-Prediction/master/Images/09%20-%20Gradient%20Descent%201D.png" width="600">


### 2D Gradient Descent
<img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Database-Prediction/master/Images/10%20-%20Gradient%20Descent%202D.png" width="300">


### 3D Gradient Descent
<img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Database-Prediction/master/Images/11%20-%20Gradient%20Descent%203D.png" width="600">






# Reinforcement Learning (Stochastic Gradient Descent)


This method is faster & more accurate than Batch Gradient Descent.



In order to avoid the Local Minimum trap, we can take more sporadic steps in random directions to increase the likelihood of finding the Global Minimum. We can achieve this by adjusting weights one row at a time (Stochastic Gradient Descent) instead of all-at-once (Batch Gradient Descent). Read 'Neural Network in 13 lines of Python.pdf' in the research papers section.



<img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Database-Prediction/master/Images/12%20-%20Local%20Min%20Trap.png">



These are the steps for Stochastic Gradient Descent:
1. Initialize weights to small numbers close to 0 (but NOT 0)
2. Input first row of Observation Data into input layer
3. Forward-propagate: Apply weights to inputs to get predicted result 'y_hat'
4. Compute Error = 'y_hat' - 'y_actual'
5. Back-propagate: Update weights according to the Learning Rate and how much they're responsible for the Error.
6. Repeat steps 1-5 after each observation (Reinforcement Learning), or after each batch (Batch Gradient Descent)
7. Epoch is the Training Set passing through the Artificial Neural Network, more Epochs yield improved results.





# Evaluating the ANN


Be careful when measuring the accuracy of a model. Bias and Variance can differ every time the model is evaluated. To solve this problem, we can use K-Fold Cross Validation which splits the data into multiple segments and averages overall accuracy.



<img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Database-Prediction/master/Images/13%20-%20Bias-Variance%20Tradeoff.png" width="400">



<img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Database-Prediction/master/Images/14%20-%20K-Fold%20Cross%20Validation.png" width="400">






# Overfitting


Overfitting is when your model is over-trained on the Training Set and isn't generalized enough. This reduces performance on Test Set predictions.



Indicators of overfitting:

1. Training and Test Accuracies have a large difference
2. Observing High Accuracy Variance when applying K-Fold Cross Validation



Solve overfitting with "Dropout Regularization", this randomly disables Neurons through iterations so they don't grow too dependent on each other. This helps the Neural Network learns several independent correlations from the data.





# Sample Problem - Bank Database Prediction


Let's test our knowledge of Artificial Neural Networks by solving a real world problem. Take a look at 'Bank_Customer_Data.csv' in the Data folder of this repository. This technique can be applied to any or any customer oriented business data set.



### Problem Description:

A Bank (or any business) is trying to improve customer retention. The Bank engineers have put together a table of data about their customers (Name, Age, Location, Income, etc). They also have data on whether customers left the Bank or stayed with them (last column of data).



The Bank is trying to build a Machine Learning model that predicts the likelihood of a customer leaving before it actually happens so they can work on improving customer satisfaction.





### Code


You can run the code online with Google Colab which is web based and doesn't require installations. 



The better alternative is to download the code and run it with 'Spyder' found in the [Anaconda Distribution](https://www.anaconda.com/download/). 'Spyder' is similar to MATLAB, it allows you to step through the code and examine the 'Variable Explorer' to see exactly how the data is parsed and analyzed.



```shell
$ git clone https://github.com/AMoazeni/Machine-Learning-Database-Prediction.git
$ cd Machine-Learning-Database-Prediction
```





# Prediction Models in Forex Markets
In this study, A. Baldo proposes a deeper analysis on the algorithmic treatment of financial time series, with a focus on Forex markets' applications. The relevant aspects of the paper refers to a more beneficial data arrangement, proposed into an image format and to the application of a Temporal Convolutional Neural Network model, representing a more than valid alternative to Recurrent Neural Netowrks. The results are supported by expanding the comparison to other more consolidated deep learning models, as well as with some of the most performing Machine Learning methods. Finally, a financial framework is proposed to test the real effectiveness of the algorithms.
## Structure of the code
The main notebook is <strong>TimeSeriesConversion.ipynb</strong> where all the experiments referring to the test models were held. This module is divided in three main sections:		
- Overview of data		
- Deep Learning models
- Machine Learning models
- Trading Strategy

The notebook <strong>DatasetCreation.ipynb</strong> shows how data are generated starting from the raw csv files present in the folder "Data"

The notebook <strong>MLModelsTuning.ipynb</strong> includes the process of the tuning of the hyperparameters of the ML models and the consequent saving of the models

The folder "Data" contains the raw data as were retrieved from the EaForexAcademy open source website. It contains exactly 11 sub-folders, referred to 11 well-noted currency pairs. Each one of this sub-folder has 7 different csv files of referred to same pair, referring to the following timeframes: 1min, 5min, 15min, 30min, 1h, 4h, 1day

The folder "DataReady" contains the prepared data as used by the different models. The division of the pairs and the files is the same as described for the "Data" folder

The folder "Models" is divided in 4 sub-folders: "CNN","RNN","TCN","ML". In this folder the all the best model weights and trained structures are saved. For the deep learning models, the architecture is saved in a JSON file, while the model parameters are in the h5 format. The ML models are in compressed in .tar.gz files. Once uncompressed, their format is .sav

The folder "Results" contains only a part of the logs of the training procedures done during this project. Also in this case, the logs contains a similar division as described for the Models folder. In general it is possible to find two files of result:
- ResultsMODELNAME.txt: contains an example of logs of result of a training procedure of the model MODELNAME on a singular currency pair
- ResultsMODELNAMEAll.txt: contains the logs of the final result of the model MODELNAME on all the currency pairs
It is also present a file "TradeResults.txt" containing the logs of the 4000 simulations of the trading strategy, done on the 8 different models

The folder "Scripts" contains the basic scripts working on network structures, data preparation, error metrics and trading strategy. In particular:
- <strong>CNN.py</strong> contains the code for the creation, training, saving and loading of the Convolutional Neural Network adopted in the project
- <strong>CustomLosses.py</strong> contains the definition of the custom loss functions, both for the Keras and PyTorch libraries
- <strong>DTW.py</strong> and <strong>FastDTW.py</strong> contain the functions and classes for the evaluation of the Dynamic and Fast Dynamic Time Warping metrics, respectively
- <strong>IndicatorsFunction.py</strong> is the code used for the evaluation of all the Technical Indicators used in project
- <strong>RNN.py</strong> contains the code for the creation, training, saving and loading of the Long Short-Term Memory Network adopted in the project
- <strong>TCN.py</strong> contains the code for the creation, training, saving and loading of the Temporal Convolutional Neural Network adopted in the project
- <strong>Trade.py</strong> is the script of the Trading Strategy
- <strong>utils.py</strong> contains the basic functions for data preparation

The folder "Trials" collect some scripts and notebooks referring to some experimental setup, mainly dealing with the Deep Learning framework. In particular:
- <strong>BayesianRNN.py, BayesianRNN.ipynb</strong> represented some experiments about a Bayesian framework on the LSTM
- <strong>TCNConfidence.py, TCNConfidenceTest.ipynb</strong> are referred to a framework where the predictions of the TCN were re-used for the subsequent predictions
- <strong>DifferentiatedData.ipynb, TCNDifftest.ipynb</strong> were some tests experimenting the benefits of differentiated time series
- <strong>TCN1D.py,TCN2D.py</strong> are customized code trying to adapt the TCN from a 1D to a 2D case (however there are some limitations of the Keras library in the 2D case)
- <strong>CNNPivot.py</strong> is a Convolutional Neural Network for the classification of the "pivot" points, where the financial trend changes its own direction
- <strong>CNNTorch.py</strong> contains the code for the creation, training, saving and loading of a Convolutional Neural Network using the PyTorch library and implementing some custom losses

## Main Libraries
All the working  models were written with the Tensorflow 2.0 and Keras Libraries
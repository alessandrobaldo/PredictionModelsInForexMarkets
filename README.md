# Prediction Models in Forex Markets

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

# Overview of the Project
[

In this study, A. Baldo proposes a deeper analysis on the algorithmic treatment of financial time series, with a focus on Forex markets’ applications. The relevant aspects of the paper refers to a more beneficial data arrangement, proposed into an image format and to the application of a Temporal Convolutional Neural Network model, representing a more than valid alternative to Recurrent Neural Netowrks. The results are supported by expanding the comparison to other more consolidated deep learning models, as well as with some of the most performing Machine Learning methods. Finally, a financial framework is proposed to test the real effectiveness of the algorithms.

]

Introduction
============

The idea of the project was suggested after the analysis of the study-case , where the authors presented some advantages in condensing 1D-time series, into 2D data formats enhancing the spatiality.
Starting from that suggestion, this project has the scope to more deeply investigate this paradigm, applying it to a regression domain. The explored environment has been the Forex markets , which are widely recognised to be very liquid and volatile markets, and thus more challenging for the algorithmic applications.
After a short explanation on the nature of data, the paper will explore a large variety of techniques. A net distinction will be presented, separating the Deep Learning framework from the Machine Learning counterpart, having here mostly a validation role on the results.
The discussion will then switch into the numerical results, examining both from an algorithmic and a financial points of view the feasibility of these models.
Finally, the last section will propose some suggestions based on the final results, as some key points that could be improved in the future to reach still more consistent results.

Forex Markets
-------------

The Forex exchange market (Forex, or FX) is the global market for the currencies and cryptocurrencies. The daily turnover is in average 6.6 trillion dollars per day[1], making it an unique market in terms of volumes and liquidity. The geographical decentralization, the possibility to trade along the all 24-hours period, weekends excluded, and the possibility to use sophisticated tools as the leverage have allowed the Forex Market to become a pool of many different profiles of traders, which are mainly distinguished in these 6 main categories: **Scalpers** are short-term traders focusing on holding positions for timeframes as small as a few seconds to a few minutes. Forex scalping strategies involve trading frequently throughout the day, with the intention of achieving small gains at the busiest (most liquid) times.
**Day Traders** also execute frequent trades on an intra-day timeframe. While their routine will not be as fast-paced as a scalper’s, they will similarly close all the positions before the end of the trading day, so as not to hold any overnight. This means trades are not affected by negative news that can hit prices before the market opens or after it closes.
**Swing Traders** hold onto trades for longer than a single day, and up to perhaps a couple of weeks. Over this short timeframe, swing traders will typically favor technical analysis over fundamentals, although they should still be attuned to the news events that can trigger volatility.
**Position Traders** hold trades for longer periods of time, from several weeks to years. As the longest holding period among trading styles, position traders are less interested in an asset’s short-term price fluctuations and more concerned, naturally, with the performance over more sustained timeframes.
**Algorithmic Traders** rely on computer programs to place trades for them at the best possible prices. Traders can use defined instructions, or high-frequency trading algorithms, to either code the programs themselves, or purchase existing products.
**Event-driven Traders** look to fundamental analysis over technical charts to inform their decisions.
**Appendix Forex** majorly describes the Forex market and the basic trading rules related to it.

Data
====

Data has been retrieved from the open-data directory EaForexAcademy [2], by collecting the historical trends of 11 major currency pairs, under 7 different time-frame intervals (from a 1 minute frequency, counting about 200’000 records, to a daily time division, mapping the last 13 years).
These raw data contains the *Open*, *Close*, *High*, *Low* prices and the *Volume* reference, from which the 28 technical indicators were computed for each interval of the series. The list of the indicators comprehends two indexes related to the variations of price and volumes between consecutive time frames, four types of moving averages, mapped into four different periods and statistical oscillators. In **Appendix Indicators** each of these features is more deeply explored and illustrated. Then, in **Appendix Feature Engineering**, the focus is moved to the engineering process behind this choice of measures.
The consequent step for the management of data dealt with the normalization of the features, due to some existing diversities. The generation of the images then collected all these records and features, at step of 28 time-frames. The target was set as the *Close* price of the immediately subsequent interval. This process was iteratively repeated, by scaling each 28 intervals window by one step forward. Some data augmentation (**Appendix Data Augmentation**) was performed to improve the model robustness.
For the training purposes, the daily time interval was considered as a trade-off between the enlargement of training times and the inclusion of data characterized by higher variances. The latter was particularly beneficial to improve the generalization capabilities of the models on shorter times frames, where the variability resulted to be more distributed in time.
The division among training, validation and test sets was respectively done according to the 72-8-10 proportions for the Deep Learning models and according to the Pareto rule (i.e. 80-20) for the Machine Learning models. As suggested by the choice of the time intervals, the focus mainly dealt with Day Traders and Scalpers scenarios.

Models
======

The model at the core of the project is the *Temporal Convolutional Network* (TCN) , representing a quite novel structure in the Deep Learning field. Due to its strict connection and derivation with the *Convolutional Neural Networks* (CNN) and the *Recurrent Neural Networks* (RNN), the project also includes an internal analysis with their performances, presenting a standard CNN and a *Long Short-Term Memory Network* (LSTM).
Finally, an external comparison is proposed between the adoption of a Deep Learning and a Machine Learning approach ,, including then some of the most performing and traditional ML techniques.

Temporal Convolutional Neural Network
-------------------------------------

Temporal Convolutional Networks, when firstly introduced in 2018, proved to have great performances on sequence-to-sequence tasks like machine translation or speech synthesis in text-to-speech (TTS) systems.
By merging the benefits of the Convolutional Neural Networks, with the memory-preserving features of the Recurrent-LSTM networks, such structures end to be more versatile, lightweight and faster than the two derivation structures.
It is desgined around the two basic principles of *causal convolution*, where no information is leaked from past to future, keeping a memory of the initial “states”, and of an architecture mapping dynamically input and output sequences of any length, allowing both the one-to-one, one-to-many and many-to-many paradigms.
Since a simple causal convolution has the disadvantage to look behind at history with size linear in the depth of the network (i.e. the receptive field grows linearly with every additional layer), the architecture employs convolutions with *dilation* (Figure [fig:dilation]), enabling an exponentially large receptive field. 
Using larger dilation enables an output at the top level to represent a wider range of inputs, thus effectively expanding the receptive field of a CNN. There are thus two ways to increase the receptive field of a TCN: choosing lager filter sizes and increasing the dilation factor.

### Residual (Temporal) Block

The Residual (Temporal) Block is the fundamental element of a TCN. It is indeed represented as a series of transformations, whose output is directly summed to the input of the block itself. Especially for very deep networks stabilization becomes important, for example, in the case where the prediction depends on a large history size with a high-dimensional input sequence.
Each block is composed of two layers of dilated convolutions and rectified linear units (ReLUs). The block is terminated by a Weight Normalization and a Dropout . Figure [fig:residualblock] shows the structure.

![residualblock](https://user-images.githubusercontent.com/48285797/104187077-a25e4480-5417-11eb-8542-2b456ce343b1.jpeg)

In **Appendix TCN** the final network structures, as well as the choice of the hyper-parameters are treated more extensively.

Recurrent LSTM Neural Network
-----------------------------

Recurrent NNs have been the standard for many years in prediction tasks of time series ,. The main idea behind is to introduce *recurrent states* to maintain an overall memory of the system and to process variable length sequences of inputs.
RNNs can have additional stored states, and the storage can be under direct control by the neural network. The storage can also be replaced by another network or graph, if that incorporates time delays or has feedback loops. Such controlled states are referred to as gated state or gated memory, and are part of Long Short-Term memory networks (LSTMs) and gated recurrent units.
Basic RNNs are a network of neuron-like nodes organized into successive layers. Each node in a given layer is connected with a one-way directed connection to every other node in the next successive layer. Each neuron has then a time-varying real-valued activation. Each connection (synapse) has a modifiable real-valued weight. Figure [fig:rnnbasic] shows the unfolded structure of a RNN.

![RNNBasic](https://user-images.githubusercontent.com/48285797/104187074-a1c5ae00-5417-11eb-9efa-5fc6b2aeaca1.png)

LSTM is a variant of a RNN introduced to cope with the vanishing gradient problem. It is usually characterized by recurrent gates called *forget gates*. In preventing the gradient to vanish or explode, LSTM networks can keep memory over longer periods. Figure [fig:lstmbasic] portrays the structure of a basic LSTM unit (i.e. LSTM cell).

![LSTMBasic](https://user-images.githubusercontent.com/48285797/104187082-a2f6db00-5417-11eb-97f9-3c775eb81edf.png)

In **Appendix LSTM** the final network structures, as well as the choice of the hyper-parameters are treated more extensively.

Convolutional Neural Network
----------------------------

The study-case also required to test a more common Convolution Network, able to exploit the spatial domain of the features. Indeed, the data format is such that similarity between the features are also enhanced by their contiguous positions in the image.
The main block of the network is thus composed of a Conv2D layer, followed by a MaxPooling2D layer and a BatchNormalization operation. The activation function is a ReLU (Rectified Linear Unit), in order to prevent overfitting.
Such block is then repeated according to the numerosity of the filters, and it is then aggregated to three Dense layers of progressively decreasing dimensions (16-4-1). The former is characterized by a ReLU activation and a Dropout operation to counterbalance overfitting, a BatchNormalization is applied too. The last two layers are instead respectively set with a ReLU and a Linear activation functions.
The final network structures, as well as the choice of the hyper-parameters are treated more extensively in **Appendix CNN**.

Linear and Bayesian Ridge Regression
------------------------------------

Linear Regression experiments were mainly held to provide a basic example of the non-linearity of financial trends. Features are thus only used in a 28 linearly-combined terms equation, where weights are adequately chosen to enhance dominant features.
For the sake of completeness, some higher order models were tested, through the use of the Polynomial Features, however their outcome is not furtherly discussed, due to the inconsistencies of the predictions.
Bayesian Ridge Regression is an algorithm based on the same framework of Linear Regression, which includes some sort of form of regularization. If compared to the OLS (Ordinary Least Squares) estimator, the coefficient weights are slightly shifted toward zeros. The prior on the weights is a Gaussian prior and the estimation of the most suitable model is done by iteratively maximizing the marginal log-likelihood of the observations.
The objective of the experiment was then to enlarge the Linear Regression spectrum with a more robust version than the simple OLS framework.

Support Vector Regression
-------------------------

Support Vector Regression (SVR) was adopted for its very well-known capability of maximizing data separation. Given the highly noise nature of data, its soft-margin version was tested under the entire diversity of kernels, in order to find the most correct feature mapping.

Random Forest Regression
------------------------

Random Forest Regression was included to enlarge the spectrum of the methods to the Ensemble learning branch and due to its high potentialities . In addition, it offers a built-in tracking method of the percentage of variance explained by each one of the features, expanding its application to the Feature Engineering (**Appendix Feature Engineering**) task. The results of this tracking method are discussed in **Appendix Random Forest**.

K-Nearest Neighbours Regression
-------------------------------

K-NN Regressors were adopted as a totally different learning paradigm. By performing a one-to-one comparison with the *K* most similar images, this method encloses and summarizes the concept of periodicity of some financial patterns from past to future. The most similar images will then highly influence the prediction for the subsequent interval.

Appendices
=====================

Details: Forex Market
=====================

Due to the exchanged volumes and its liquidity, Forex markets require an high-precision tracking of the evolution of the trends. The fundamental unit of measure is the *pip*, representing the penultimate significative decimal number. In most cases, that coincides with the 4th-5th decimal number, but it is not so uncommon for it to be the at the 2nd-3rd decimal place. Indeed, pairs where the inflation of the quoted currency is very high are more frequently characterized by a minor precision. The exotic currencies fell in this scenario, having less market power when compared to the occidental ones. Also for this reason, trading over these pairs is often restricted to well-navigated traders who can cope with high volatility scenarios.
However, even though the variations on the price seem to be quite small, large profits and losses in Forex are commonplace, due to its highly noisy and stochastic nature. Furthermore, the quantities required to investors to open a position in the market are often very high and are denoted by the measure called *lot*. A lot is equivalent to 100’000 times the fundamental unit of the quoted currency (e.g., investing one lot over the EURUSD currency pair means investing 100’000\$). The minimum lot size a trader can choose to invest is 1/100 of the lot unit.
In order to make these amounts available to every trader, the use of leverage is very widespread in Forex markets, reaching multiplicative factors of 100-500x. The additional amount of money required by a leveraged investment is ensured by the broker figure a trader would rely to. A broker can be either a centralized or decentralized entity which establishes its own conditions on each trader account. Normally, brokers do not apply fees on the transactions (except for maintaining open the orders over the night), since their major source of gain is related to the small existing differences between the *bid* and *ask* prices, called *spreads*
If on the one hand a trader aiming to enter the market is facilitated by these mechanisms, on the other hand there are some forms of guarantees for the brokers which are limiting. A broker has in fact the possibility to automatically close lossy investments, whether a singular investment or a collection of them makes the budget drop below a determined margin. This event is often anticipated by a *margin call*, in order to guarantee transparency towards the trader.
Market orders can be either BUY or SELL orders, according if the trader expects the market to be bullish or bearish. Orders can be then opened directly or be pending, whether the trader sets some conditions the price has to meet before opening them. When an order is opened, a trader would normally set the Take Profit (TP) and Stop Loss (SL) thresholds, according to the profit he/she expects to achieve and to the risk he/she is intended to be exposed. In the case of BUY (resp. SELL) orders, the TP (resp. SL) is set above the entry price, while the SL (resp. TP) is set below.
Finally, in the short term, Forex markets are often analysed by the traders by making large use of technical analysis more than fundamental analysis, consisting of a lot of statistics and technical indicators.

Details: Indicators
===================

-   **Yield**

    ![Yield](https://user-images.githubusercontent.com/48285797/104350672-b469e100-5504-11eb-8e50-c2320923b703.png)  

-   **Percentage Volume**
    
    ![PercVol](https://user-images.githubusercontent.com/48285797/104350666-b338b400-5504-11eb-9bea-2925facbe4e7.png)
    
-   Simple Moving Averagee (**SMA**) 
    
    ![SMA](https://user-images.githubusercontent.com/48285797/104350667-b3d14a80-5504-11eb-941d-fe00ef658433.png)
    
-   Exponential Moving Averages (**EMA**)
    
    ![EMA](https://user-images.githubusercontent.com/48285797/104350661-b2a01d80-5504-11eb-95d8-092d66a6ab7d.png)
   
-   Weighted Moving Averages (**WMA**) 
    
    ![WMA](https://user-images.githubusercontent.com/48285797/104350671-b469e100-5504-11eb-89c1-1b8b8ebdfd10.png)
    
-   Hull Moving Averages (**HMA**)
    
    ![HMA](https://user-images.githubusercontent.com/48285797/104350669-b3d14a80-5504-11eb-890e-05da4911988b.png)

-   Moving Average Convergence/Divergence (**MACD**)
    
    ![MACD](https://user-images.githubusercontent.com/48285797/104350676-b5027780-5504-11eb-9986-04662ccc24c6.png)
    
-   Commodity Channel Index (**CCI**)
    
    ![CCI](https://user-images.githubusercontent.com/48285797/104350674-b5027780-5504-11eb-8391-078f45eea342.png)

-   **Stochastic Oscillator**
    
    ![StochOsc](https://user-images.githubusercontent.com/48285797/104350662-b2a01d80-5504-11eb-8c68-4b747eeba6ee.png)

    where *H14,L14* are repsectively the Highest and Lowest prices registered in the last 14 time intervals

-   Relative Strength Index (**RSI**)
    
    ![RSI](https://user-images.githubusercontent.com/48285797/104350660-b2078700-5504-11eb-9957-e7551f0a9e4f.png)

    where *U,D* are respectively the average of the differences *Close-Open* of the last *n* Bullish/Bearish bars

-   Rate of Change (**ROC**)
    
    ![ROC](https://user-images.githubusercontent.com/48285797/104350655-b16ef080-5504-11eb-9d2f-7ac53952edf1.png)

-   Percentage Price Oscillator (**PPO**)
    
    ![PPO](https://user-images.githubusercontent.com/48285797/104350656-b2078700-5504-11eb-9f6a-63908fd6896c.png)

-   Know Sure Thing (**KST**)
    
    ![KST](https://user-images.githubusercontent.com/48285797/104350665-b338b400-5504-11eb-9442-029952dad828.png)

-   Bollinger Bands Middle, Up and Down (**BOLM**,**BOLU**,**BOLD**)
    
    ![BOLS](https://user-images.githubusercontent.com/48285797/104350651-b0d65a00-5504-11eb-8cec-5c2b1103c918.png)

    where ![sigma](https://user-images.githubusercontent.com/48285797/104351349-92249300-5505-11eb-89e8-dd23fd142a84.png) is the standard deviation of the last 20 Typical Prices

Feature Enginnering
===================

Correlation between Features
----------------------------

Correlations measures at Figure [fig:corr1] highlight the two different natures among the features: the Moving Averages family and the Oscillators. The former presents a flat behaviour, comprehending also the Bollinger Bands (which exploit themselves a concept of Moving Average); the latter category is more diversified, underlining not only the similarities between the RSI and the Stochastic Oscillators (which map conceptually the same information), but also evidencing the importance to include an indicator like the Know Sure Thing (KST), apparently characterizing in a different way the evolution of the price.

![CorrFeatures](https://user-images.githubusercontent.com/48285797/104187092-a4280800-5417-11eb-8258-34586152f4ab.png)

Correlation between Features and Labels
---------------------------------------

The Moving Averages are broadly adopted for their forecasting capabilities and they are often used singularly as predictors in regression tasks. According to this, the correlation measures reported at Figure [fig:corr2] are coherent with the nature of those features.

![CorrFeaturesLabel](https://user-images.githubusercontent.com/48285797/104187091-a4280800-5417-11eb-93ae-63a44188328f.png)

Mutual Information of the Features and Motivations
--------------------------------------------------

The Mutual Information represents a standard metric in Machine Learning to perform Feature Engineering. If the correlation measures give a global knowledge on the features, more specific statistics are useful to explain the contribution of each information in the final prediction. Table [tab:mutualinfo] furtherly confirms the central role of the Moving Averages in the predictions, evidencing how for the intra-daily scenario the fastest moving averages (i.e. the moving averages with a smaller window) are more adequate.

![Mutual](https://user-images.githubusercontent.com/48285797/104189568-2403a180-541b-11eb-9056-e7d69c0d057b.png)

However, the support of slower MAs is justified by many financial strategies. Indeed, the financial technical analysis often makes use of combinations of them, by varying the type and the period. A longer period is usually referred to a slow moving average, because the indicator better absorbes variability by averaging it on the last *period* timeframes. On the contrary, a faster moving average is associated to a shorter averaging period, being thus more influenced by short-term variability.
Traders generally look to the crossovers of slow and fast moving averages, since they create some accurate pivot points, useful to determine the entry or exit points for BUY/SELL orders.
The Oscillators, instead, mostly track the different natures of the so called *Momentum*. This has the equivalent meaning of the instantaneous speed in physics, evaluating the price variation speed for a defined period. In addition, indicators like the RSI and the Stochastic Oscillators are frequently used to classify Overbought and Oversold regions in the trend. The main principle is that each trend should have an implicit equilibrium overall, untethered by the bullish or bearish global nature of the market. Therefore, very rapid variations would unbalance the system and these tools are capable to track when it is likely the trend will undergo to a movement in the opposite direction. Conventionally, the period set for these indicators is around 12-14 days.

Data Augmentation
=================

Data Augmentation represents a form of pre-processing which often grants to improve the overall model’s robustness. For the sake of this project, it consisted in randomly selecting a pre-defined percentage of images (about 30%) and applying on them an “obscuration” (i.e. totally nullifying some image rows) of some of data closer in time to the referred label. In this way, the model was forced to rely more on the data references further in time, avoiding to simply output a prediction based on the last few recorded references, where an high autocorrelation with the label existed.
Other forms of data augmentation can be furtherly thought, by including for example some samples from other correlated currency pairs, but that went out of the scopes of this project.

Model Details
=============

As a baseline, all the deep learning models were implemented through the Keras library and were tuned by using some built-in callbacks, allowing for a dynamic reduction of the learning rate over the epochs, when a plateau was detected and which opportunely stopped the training if some overfitting was occurring.

Temporal Convolutional Network (TCN)
------------------------------------

The final TCN structure was conceived by correctly testing different combinations of layers and depth of the convolutional layers. The overall network is reported at Figure [fig:TCN] and resulted in a 7-levels network, where each one is represented by a Residual (Temporal) Block. The high number of filters of each convolutional layer combined with the high dilation factor gave the network optimal memory-preserving properties.

![TCNFinal](https://user-images.githubusercontent.com/48285797/104187068-a0948100-5417-11eb-9952-92b2dce6a4c8.png)

The other optimized parameters used during the experiments were:

-   Epochs: 200

-   Learning Rate: 0.01

-   Batch Size: 32

-   Kernel Size: 3

Long Short-Term Memory Network (LSTM)
-------------------------------------

For the final network configuration, the number of LSTM layers and the number of cells inside each layer were used as hyperparameters to be tuned; in the end, the final structure (Figure [fig:LSTM]) had the following settings:

-   LSTM layers: 2

-   LSTM Cells per layer: 128

-   Epochs: 200

-   Learning Rate: 0.01

-   Batch Size: 32

![LSTMFinal](https://user-images.githubusercontent.com/48285797/104187081-a2f6db00-5417-11eb-82e4-f1a1e55c600b.png)

Convolutional Neural Network (CNN)
----------------------------------

As previously mentioned, also the CNN was tuned by including the disposition of the elements of the architecture as hyperparameters. In this case, both the number of convolutional blocks and filters, as well as the kernel, were taken into account. In the end, the structure resulted to be as Figure [fig:CNN] portrays, having the following setup:

-   Epochs: 200

-   Learning Rate: 0.005

-   Batch Size: 32

-   Kernel size: 7

![CNNFinal](https://user-images.githubusercontent.com/48285797/104187093-a4c09e80-5417-11eb-8c07-c40c91690d2b.png)

Random Forest Regression and Impurity 
--------------------------------------

The Random Forest Regression models allowed to evaluate further statistics on the explained variance of each future on the final prediction. This can be done by the model by monitoring how the OOB (out-of-the-bag) error evolves according to the splits at each node and the features excluded by each estimator. As Table [tab:explainedvar] summarizes, over 95% of the total variance is explained by 5 of the fastest moving averages, being another confirmation of the aforementioned Feature Engineering.

![Variance](https://user-images.githubusercontent.com/48285797/104189563-236b0b00-541b-11eb-84d1-0b159c476a4a.png)

Error Measures: Pearson Correlation, Dynamic Time Warping (DTW) and Fast DTW
============================================================================

The first measure the author decided to introduce was a correlation term, which was able to correctly capture the general trend of the true time series. The Pearson Correlation ([eq:pearson]) thus represented an as simple as effective mathematical indicator for what concerns the spatial evolution of the financial trend.

\begin{equation}\label{eq:pearson}
    \rho_{XY}= \frac{cov(X,Y)}{\sigma(x)\sigma(Y)}
    \end{equation}

In the deep learning experiments it was firstly used retrospectively as a method to favour those models which not only minimized the distance with respect to the real trend (i.e. low MSE), but also which were able to maximize the correlation. The Equation ([eq:acceptance]) models the decision rule adopted at the end of each training procedure:

\begin{equation}\label{eq:acceptance}
    \underset{\theta}{\mathrm{argmin}}~ \mathrm{MSE}_{\hat{y},y} \cdot (1-\rho_{\hat{y},y})
    \end{equation}

This simple mathematical combination ensured to have a more global view on the outcomes and to increase the overall optimality.
The next tentative mainly dealt with the plug-in of such decision rule inside the loss function itself, but, due to the consequent increase of complexity and training times of the networks structure, this task was left for future improvements.
Given the aforementioned spatial modelling, the intent was then to include in the error measures an indicator which could symmetrically act on the temporal domain. Indeed, when dealing with random walk time series (as the financial case is), predictive models are distinguished by systematically missing the so-called *pivot* points, i.e. where the trend changes its concavity (on the local minima and maxima). This often results in a delay in the prediction of these points, since each model prediction would be obtained based on the high autocorrelation existing with the last known sample.
In literature ,, there are several studies about the most correct temporal measures/loss to introduce in deep learning models and in this project two of them have been adopted to verify whether an increase of the complexity is justified with respect to their effectiveness.
The first measure is the so-called *Dynamic Time Warping*. Behind the computation of the DTW there is a solution of a Quadratic Optimization Problem making use of Dynamic Programming. The value of the indicator is the value of the shortest path built between two time series, according to a *window* parameter, which regulates a one-to-one unidirectional mapping from the predicted output and the real trend.
Despite its effectiveness in describing the temporal nature of the predictions, its complexity (\(O(n^2)\)) is quite limiting, especially whether there is the necessity to plug it into a loss function. As did with the Pearson correlation coefficient, its evaluation was done retrospectively according to the following updated decision rule:

\begin{equation}
\underset{\theta}{\mathrm{argmin}}~ \mathrm{MSE}_{\hat{y},y} \cdot (1-\rho_{\hat{y},y})\cdot \mathrm{DTW}_{\hat{y},y, window}
\end{equation}
As stated before, some brief experiments were held by customizing the loss function used in deep learning models, but that would have been too time consuming.
The last indicator reported for the sake of completeness is the Fast-DTW (*Fast Dynamic Time Warping*) . It is a simplified, but equally accurate version of the DTW, bounding the calculations to a linear complexity.

Details: Trading Strategy
=========================

The trading strategy is an event-based simulation, automatically deciding whether to enter or not the market for each time $t$, based on the price evolution predicted at $t+1$. In doing so, it is checked if the current predicted price \(\tilde{p}(t)\) represents a local minimum (i.e. $\tilde{p}(t-1)>\tilde{p}(t)<\tilde{p}(t+1)$) or maximum (i.e. $\tilde{p}(t-1)<\tilde{p}(t)>\tilde{p}(t+1)$ ) to then open a BUY or a SELL position, respectively.
In opening an order, of course, the real price is considered as the entry price. A Take Profit (TP) threshold is set by considering the predicted delta. A Stop Loss (SL) value is determined too, based on the $risk Factor$ parameter the strategy is configured. This parameter regulates the predisposition to risk of the investor: the higher the \(risk Factor\), the minor the risk, and thus the closer will be the SL with respect to the TP threshold. Also in these cases, both the TP and SL values are referred to true prices.
The trading method then defines a $budget$ value as the initial capitalization and a $margin Call$ value, simulating the notification a broker would take in case the investments became too lossy. In this case, the strategy automatically limits the possibility to invest large quantities.
These quantities are also governed by $leverage$ parameter, which automatically allows to determine an upper bound on the lot measure, represented by the $maximumLot$ parameter. This is continuously updated by taking into consideration the available margin (i.e. the free capitalization quota of the budget above the $margin Call$) and the adopted leverage. The so-limited $lot$ measure to be invested will be then calculated according to the outcome of the previous transactions: profitable transactions will allow to increase it by a \(step Lot\) measure, while lossy investments will downgrade the previous \(lot\) value of two steps (i.e. in this sense, by penalizing more the prediction errors of the algorithms, this can be thought as another countermeasure towards risk).
Finally the trading strategy accepts a $smooth Factor \in (0,1]$ parameter, which determines how much confidence the trader wants to give to the predictor: by choosing a low $smooth Factor$ the trader would pursue a more conservative approach, by smoothing the predictions of the variations and thus limiting the decisions of the model.
After an order is opened and the available margin is decreased, at each subsequent time interval it is monitored by coherently increasing/decreasing the budget according to how the true price evolved; eventually, the order is closed if the Take Profit (respectively the Stop Loss) was met in the transition from the previous time frame. To do so, the High and the Low prices of the time frames are considered to check if the extreme variations of the price overcame one of the two thresholds.
Despite the very detailed implementation of this automatic trading strategy, the algorithm has to cope with the limits of the time granularity. It means that the algorithm is not able to determine if, in a time interval where the real price met both the TP and SL, it had hit before one than the other. In order to accomplish a worst case scenario (i.e. the minimum achievable profit), in such cases the algorithm always registers a loss. In general, this decision process happens quite often (about 30% of the total market orders is characterized by this ambiguity), therefore the results are still more pessimistic than the real materializable worst situation.


![RandomForest](https://user-images.githubusercontent.com/48285797/104187079-a2f6db00-5417-11eb-955f-c89f962ecbd2.png)



![TCN](https://user-images.githubusercontent.com/48285797/104187070-a12d1780-5417-11eb-90fd-be445163e433.png)



![RNN](https://user-images.githubusercontent.com/48285797/104187076-a25e4480-5417-11eb-96f3-807d5120a363.png)



[1] Data retrieved in April 2019 by the Bank for International Settlement

[2] https://eaforexacademy.com/software/forex-historical-data/



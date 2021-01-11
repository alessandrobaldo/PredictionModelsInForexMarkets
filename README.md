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

# Overview
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
**Appendix [app:forex]** majorly describes the Forex market and the basic trading rules related to it.

Data
====

Data has been retrieved from the open-data directory EaForexAcademy [2], by collecting the historical trends of 11 major currency pairs, under 7 different time-frame intervals (from a 1 minute frequency, counting about 200’000 records, to a daily time division, mapping the last 13 years).
These raw data contains the *Open*, *Close*, *High*, *Low* prices and the *Volume* reference, from which the 28 technical indicators were computed for each interval of the series. The list of the indicators comprehends two indexes related to the variations of price and volumes between consecutive time frames, four types of moving averages, mapped into four different periods and statistical oscillators. In **Appendix [app:indicators]** each of these features is more deeply explored and illustrated. Then, in **Appendix [app:featureeng]**, the focus is moved to the engineering process behind this choice of measures.
The consequent step for the management of data dealt with the normalization of the features, due to some existing diversities. The generation of the images then collected all these records and features, at step of 28 time-frames. The target was set as the *Close* price of the immediately subsequent interval. This process was iteratively repeated, by scaling each 28 intervals window by one step forward. Some data augmentation (**Appendix [app:dataaug]**) was performed to improve the model robustness.
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
Since a simple causal convolution has the disadvantage to look behind at history with size linear in the depth of the network (i.e. the receptive field grows linearly with every additional layer), the architecture employs convolutions with *dilation* (Figure [fig:dilation]), enabling an exponentially large receptive field. Under a mathematical point of view, this means mapping an input sequence \(x \in \mathcal{R}^T\) with a filter \(f:\{0,...k-1\}\rightarrow \mathcal{R}\), using a convolution operator \(F\) in this way:

\[F(x)=(x_d f)(x)=\sum_{i=0}^{k-1}x_{s-d \cdot i}\]

where \(d=2^l\) is the dilation factor, \(l\) the level of the network and \(k\) the kernel size of the filter.

![Dilated Convolution](imgs/dilation.jpeg "fig:") [fig:dilation]

Using larger dilation enables an output at the top level to represent a wider range of inputs, thus effectively expanding the receptive field of a CNN. There are thus two ways to increase the receptive field of a TCN: choosing lager filter sizes \(k\) and increasing the dilation factor \(d\), since the effective history of one layer is \((k-1)d\).

### Residual (Temporal) Block

The Residual (Temporal) Block is the fundamental element of a TCN. It is indeed represented as a series of transformations, whose output is directly summed to the input of the block itself. Especially for very deep networks stabilization becomes important, for example, in the case where the prediction depends on a large history size with a high-dimensional input sequence.
Each block is composed of two layers of dilated convolutions and rectified linear units (ReLUs). The block is terminated by a Weight Normalization and a Dropout . Figure [fig:residualblock] shows the structure.

![Residual (Temporal) Block](imgs/residualblock.jpeg "fig:") [fig:residualblock]

In **Appendix [app:TCN]** the final network structures, as well as the choice of the hyper-parameters are treated more extensively.

Recurrent LSTM Neural Network
-----------------------------

Recurrent NNs have been the standard for many years in prediction tasks of time series ,. The main idea behind is to introduce *recurrent states* to maintain an overall memory of the system and to process variable length sequences of inputs.
RNNs can have additional stored states, and the storage can be under direct control by the neural network. The storage can also be replaced by another network or graph, if that incorporates time delays or has feedback loops. Such controlled states are referred to as gated state or gated memory, and are part of Long Short-Term memory networks (LSTMs) and gated recurrent units.
Basic RNNs are a network of neuron-like nodes organized into successive layers. Each node in a given layer is connected with a one-way directed connection to every other node in the next successive layer. Each neuron has then a time-varying real-valued activation. Each connection (synapse) has a modifiable real-valued weight. Figure [fig:rnnbasic] shows the unfolded structure of a RNN.

![Unfolded RNN](imgs/RNNBasic.png "fig:") [fig:rnnbasic]

LSTM is a variant of a RNN introduced to cope with the vanishing gradient problem. It is usually characterized by recurrent gates called *forget gates*. In preventing the gradient to vanish or explode, LSTM networks can keep memory over longer periods. Figure [fig:lstmbasic] portrays the structure of a basic LSTM unit (i.e. LSTM cell).

![LSTM Unit](imgs/LSTMBasic.png "fig:") [fig:lstmbasic]

In **Appendix [app:LSTM]** the final network structures, as well as the choice of the hyper-parameters are treated more extensively.

Convolutional Neural Network
----------------------------

The study-case also required to test a more common Convolution Network, able to exploit the spatial domain of the features. Indeed, the data format is such that similarity between the features are also enhanced by their contiguous positions in the image.
The main block of the network is thus composed of a Conv2D layer, followed by a MaxPooling2D layer and a BatchNormalization operation. The activation function is a ReLU (Rectified Linear Unit), in order to prevent overfitting.
Such block is then repeated according to the numerosity of the filters, and it is then aggregated to three Dense layers of progressively decreasing dimensions (16-4-1). The former is characterized by a ReLU activation and a Dropout operation to counterbalance overfitting, a BatchNormalization is applied too. The last two layers are instead respectively set with a ReLU and a Linear activation functions.
The final network structures, as well as the choice of the hyper-parameters are treated more extensively in **Appendix [app:CNN]**.

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

Random Forest Regression was included to enlarge the spectrum of the methods to the Ensemble learning branch and due to its high potentialities . In addition, it offers a built-in tracking method of the percentage of variance explained by each one of the features, expanding its application to the Feature Engineering (**Appendix [app:featureeng]**) task. The results of this tracking method are discussed in **Appendix [app:randomforest]**.

K-Nearest Neighbours Regression
-------------------------------

K-NN Regressors were adopted as a totally different learning paradigm. By performing a one-to-one comparison with the \(K\) most similar images, this method encloses and summarizes the concept of periodicity of some financial patterns from past to future. The most similar images will then highly influence the prediction for the subsequent interval.

Results
=======

All the results reported were achieved under identical conditions. Each model underwent to a Cross Validation procedure, by computing the final score on the same test dataset constituted by more than 800 days up to the end of October 2020. Incorporating the COVID-19 pandemic financial crisis, this surely made this period the most intriguing and challenging to test the generalization capabilities of the models.
For sake of brevity, only a part of the results has been reported for an immediate comparison. In Table [tab:comparison] the eight models are referred to daily Forex data of the 11 currency pairs taken into account. The Mean Squared Error (MSE) measures represented the main benchmark for the goodness of each model. However, some further measures (**Appendix [app:errormeasures]**) and more extensive summaries of the results are subsequently reported.

[!hbtp] [tab:comparison]

Algorithmic Results
-------------------

Looking at the numerical results of the Machine Learning methods, the Random Forest Regression represented the best performing model under all the adopted metrics (Table [tab:RandomForest]).
Conversely, the SVR model did not guarantee to be as accurate as the other models did; however, it represents the evidence of why adopting multiple judgement criteria: it is indeed true that it presents the lowest MSE values, but at the same time, as Table [tab:SupportVector] proves, it achieves good correlation scores, catching both the spatiality and temporality of the trend. Another important insight in this sense can be retrieved in the two basic regression models, where the adoption of a more regularized framework (Bayesian Ridge Regression) does not seem to heavily impact neither the MSE statistics nor the Pearson correlation reached by Linear Regression (Tables [tab:LinearRegression], [tab:BayesianRidgeRegression]), though, in the temporal domain it improved the related metrics of an almost 30%.
To complete the Machine Learning models’ picture, the KNN Regressor, exploiting the similarities with 50 previous temproal snapshots (Table [tab:KNN]), is able to perform correctly in each situation. However the highly prohibitive computational cost, it could represent a discriminative factor which would surely favour the adoption of other models.

The Deep Learning counterpart was instead tested on a larger variety of datasets, including also smaller timeframes. Such a choice was primarily conceived as a further benchmark for the generalization capabilities of the models, as well as to test whether the parameters modelled on the daily frameworks could well adapt also in minor variability circumstances, as intra-daily time intervals are.
Having an overview of the situation, it is evident that the TCN (Table [tab:TCN]) and the LSTM (Table [tab:RNN]) represent the most adequate structures, if compared to the CNN. The metrics scores does not highlight any evident superiority between them, if not on the more “exotic” (and volatile) currency pairs (i.e. where the quoted currency was the Japanese Yen), where the Temporal ConvNet resulted to be more robust.
However, there are quite few reasons to prefer the latter to the standard LSTM structures: firstly, the TCN proved to have major memory-preserving properties as the depth was increased: indeed, by deepening the network with new Temporal Blocks, the model became still more performant; conversely the LSTM structure still presented some vanishing “memory” issues as the number of layers was augmented, being less robust with quite different currencies.
Secondly, the TCN is more lightweight (i.e. it has a minor number of parameters to be trained) than a corresponding LSTM with same depth, resulting in faster and easier training procedures.
Finally, due to algorithmic limitations, the adopted Temporal Convolutional Network was developed in a one-dimensional case, without exploiting the benefits deriving from data spatiality and 2-D convolutions. This is in particular a pivotal point, that could still more reflect the efficiency of such a structure.
In the end, the CNN structure resulted to be the least performant among the deep learning models, being often one order of magnitude above the other networks’ results. Moreover, especially during the COVID-19 pandemic period, the network totally failed to generalize the trend. Overall, its behaviour could be still considered positively: the CNN indeed only exploits the spatiality of data, without having the possibility to retrieve information from the time domain. This is a confirmation of what was held in about a more effective disposition of information: financial data can thus be made more informative according to how features are related to each other in the input data.

Financial Results
-----------------

[!htbp] [tab:financialresults]

Algorithmic trading tools are completely automatized procedures, whose logic should be to overcome subjectivity limitations of human traders, as well as to enter positions with promptness when a possible profit is detected. To do so, the “awareness” of such algorithms does not imply neither to continuously enter the market with new positions, nor to necessarily forecast the exact value of a trend in a time in the future.
However, the distinctive feature for a financial predictive algorithm represents the timing in the recognition of the pivot points, where the trend changes its direction. In Forex, especially, this is fundamental both to obtain full profits, by not closing early the positions, and to avoid losses. The real forecasting capability is then only partially referred to the global distance from the real trend, which sometimes could be fairly acceptable without being as explanatory as should be.The generalization capabilities of these algorithms when fine-tuned are mostly unquestionable, underlining the efficacy they could have in the financial context. On the other hand, also taking the best one of these methods in its best performing case, would not guarantee a risk-free, and thus valid, method for practical trading sessions.
A trading strategy was adopted, in order to create a procedural framework to enter the market. Its full description is inserted in **Appendix [app:tradingstrategy]**.
The tests have been performed on a over 800-days window at different levels of risk-reward ratio, with different initial budgets and leverages and with different confidences towards the algorithms. As a result, each model (**Table [tab:financialresults]**) counted a total of 4000 simulations, denoting the following patterns:
- when the strategy gives full confidence to the algorithms and opt for a poor minimization of the risk, the average outcome is in general more extreme, amplifying losses whether the model is not good enough, but also helping to achieve higher profits in shorter times. The K-NN Regressor behaviour is an example of the latter case, where the final budget resulted to be 30 times larger than the initial one, often reaching at least 500% as net return. Although this scenario seems to be quite unrealistic, given the very risky strategy this is not so uncommon neither in a real context
- when the confidence is kept reasonably low, an improvement/stabilization on the performances is registered, especially whether the model encountered many losses. When the return-risk ratio is incremented, it happens too, as well. The combinatorial effect of the two settings describes then a balanced conservative strategy: in the long-term guarantees profits on the performant models, while it ensures a total minimization of the losses, mostly absorbed by the time window
- since the strategy, by construction, deals with a scenario penalizing a lot of likely profitable market orders, these results still represent a lower bound on the worst-case scenario. Therefore we can look at them optimistically
- most of the models registered huge losses during the COVID-19 pandemic period, representing a singular and quite unexpected event, also under the financial point of view. Only the CNN had some “benefits” from this particular period, since, due to a great lack of precision in the predictions, its trading strategy mostly avoided to enter the market and, thus, the related losses.

Conclusions and Next Steps
==========================

The project explored the potentialities of Deep Learning methods in financial time series, showing internal comparisons between some of the most wide-spread techniques and innovative performing structures, and external comparisons with the Machine Learning domain. The Temporal Convolutional Network proved to be reliable, having similar performances to the Recurrent Neural Long Short-Term Memory Network and presenting advantages in terms of ease of training and scalability. Furthermore, the singular nature of data defines another “hyper-parameter” to be considered in the variety of model parameters: some improvements can indeed be retrieved by enhancing spatiality characterizations among features.
The discussion then moved to the necessity of introducing more meaningful error metrics and the possibility to plug them into the loss functions. The derived benefits of this operation could ensure better updates (and thus learning), definitely creating a performance gap between Deep Learning and Machine Learning.
From a financial point of view, we discussed how this random walk behaviour of financial time series represents the biggest obstacle to make these predictions be highly reliable and so generate automatic profits. However, as some literature remarks, a palliative could be operating a differentiation on the time series as a pre-processing, obtaining an almost stationary series.
On the other hand, the results of the simulation hopefully denote that a correct training could lead to an overall minimization of the risks and, in some cases, to profitable sessions. In addition, a singular event as COVID-19 pandemic represented, should be considered to be inserted inside the training set, in order to amplify data diversity. In general, it is difficult to discriminate whether a correct training dataset should be mostly specialized on one currency pair, or be extended to other correlated/uncorrelated pairs. This project by pursuing the former thesis, aimed at maximizing the generalization capabilities on the singular quoted currency.
Finally, further improvements should then focus particularly on the capabilities of the models to correctly detect pivot points, where the trend changes its direction. This could be done by plugging a classifier on top of the regression models, discriminating between pivot and non-pivot points.

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

    \[Yield = \frac{Close - Open}{Open}\]

-   **Percentage Volume**

    \[PercentageVolume = 10^4\frac{High - Low}{Volume}\]

-   Simple Moving Averagee (**SMA**) of period \(n\)

    \[\mathrm{SMA}_i(Close, n) = \frac 1n \sum_{i=0}^n Close_{n-i}\]

-   Exponential Moving Averages (**EMA**) of period \(n\)

    \[\begin{split}
        & \mathrm{EMA}_i(Close, n) = \\
        & \alpha\cdot Close_{i} + (1-\alpha)\cdot \mathrm{EMA}_{i-1}(Close,n)
        \end{split}\]

-   Weighted Moving Averages (**WMA**) of period \(n\)

    \[\begin{split}
           &  \mathrm{WMA}_i(Close, n) =\\
           & \frac{\sum _ {i=0}^{n-1} i \cdot Close_{i}}{\frac{n(n-1)}2}
        \end{split}\]

-   Hull Moving Averages (**HMA**) of period \(n\)

    \[\begin{split}
            & \mathrm{HMA}_i(Close, n) = \mathrm{WMA}_i(arg,\sqrt{n}),\\
            & arg =(2\cdot \mathrm{WMA}(Close,\frac n 2) - \mathrm{WMA}(Close,n))
        \end{split}\]

-   Moving Average Convergence/Divergence (**MACD**)

    \[\mathrm{MACD} = \mathrm{EMA}(Close, 12) - \mathrm{EMA}(Close, 26)\]

-   Commodity Channel Index (**CCI**)

    \[\begin{split}
            & \mathrm{CCI} =\\
            & \frac{TypicalPrice - \mathrm{SMA}(TypicalPrice,20)}{0.015 \times AvgDev},\\
            & TypicalPrice = \frac{High + Low + Close}3
        \end{split}\]

-   **Stochastic Oscillator**

    \[\begin{split}
            & StochasticOscillator =\\
            & 100\frac{Close - H14}{H14 - L14}
        \end{split}\]

    where \(H14,L14\) are repsectively the Highest and Lowest prices registered in the last 14 time intervals

-   Relative Strength Index (**RSI**)

    \[\begin{split}
            & \mathrm{RSI}(n) = 100 - \frac{100}{1+\mathrm{RS}},\\
            & \mathrm{RS} = \frac U D, \quad n=14
        \end{split}\]

    where \(U,D\) are respectively the average of the differences \(Close - Open\) of the last \(n\) Bullish/Bearish bars

-   Rate of Change (**ROC**)

    \[\begin{split}
            & \mathrm{ROC}(n) = 100 \frac{Close_i - Close_{i-n}}{Close_{i-n}},\\
            & n = 14 
        \end{split}\]

-   Percentage Price Oscillator (**PPO**)

    \[\begin{split}
            & \mathrm{PPO} =\\
            & 100 \frac{\mathrm{EMA}(Close, 12) - \mathrm{EMA}(Close, 26}{\mathrm{EMA}(Close,26)}
        \end{split}\]

-   Know Sure Thing (**KST**)

    \[\begin{split}
            & \mathrm{KST} = \mathrm{SMA}(\mathrm{RCMA}_1 + 2\mathrm{RCMA}_2 +\\
            & 3\mathrm{RCMA}_3 +4\mathrm{RCMA}_4, 9),\\
            & \mathrm{RCMA}_1 = \mathrm{SMA}(\mathrm{ROC}(10), 10),\\
            & \mathrm{RCMA}_2 = \mathrm{SMA}(\mathrm{ROC}(15), 10),\\
            & \mathrm{RCMA}_3 = \mathrm{SMA}(\mathrm{ROC}(20), 10),\\
            & \mathrm{RCMA}_4 = \mathrm{SMA}(\mathrm{ROC}(30), 15) 
        \end{split}\]

-   Bollinger Bands Middle, Up and Down (**BOLM**,**BOLU**,**BOLD**)

    \[\begin{split}
        & \mathrm{BOLM} = \mathrm{SMA}(TypicalPrice, 20)\\
        & \mathrm{BOLM} = \mathrm{BOLM}+ 2\sigma_{20}(TypicalPrice)\\
        & \mathrm{BOLM} = \mathrm{BOLM}- 2\sigma_{20}(TypicalPrice)
        \end{split}\]

    where \(\sigma_{20}(TypicalPrice)\) is the standard deviation of the last 20 Typical Prices

Feature Enginnering
===================

Correlation between Features
----------------------------

Correlations measures at Figure [fig:corr1] highlight the two different natures among the features: the Moving Averages family and the Oscillators. The former presents a flat behaviour, comprehending also the Bollinger Bands (which exploit themselves a concept of Moving Average); the latter category is more diversified, underlining not only the similarities between the RSI and the Stochastic Oscillators (which map conceptually the same information), but also evidencing the importance to include an indicator like the Know Sure Thing (KST), apparently characterizing in a different way the evolution of the price.

![Correlation Matrix among Features](imgs/CorrFeatures.png "fig:") [fig:corr1]

Correlation between Features and Labels
---------------------------------------

The Moving Averages are broadly adopted for their forecasting capabilities and they are often used singularly as predictors in regression tasks. According to this, the correlation measures reported at Figure [fig:corr2] are coherent with the nature of those features.

![Correlation Matrix between Features and Target](imgs/CorrFeaturesLabel.png "fig:") [fig:corr2]

Mutual Information of the Features and Motivations
--------------------------------------------------

The Mutual Information represents a standard metric in Machine Learning to perform Feature Engineering. If the correlation measures give a global knowledge on the features, more specific statistics are useful to explain the contribution of each information in the final prediction. Table [tab:mutualinfo] furtherly confirms the central role of the Moving Averages in the predictions, evidencing how for the intra-daily scenario the fastest moving averages (i.e. the moving averages with a smaller window) are more adequate.

[]

[tab:mutualinfo]

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

![The final TCN resulting from the hyperparameters’ tuning](imgs/TCNFinal.png "fig:") [fig:TCN]

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

![The final LSTM network resulting from the hyperparameters’ tuning](imgs/LSTMFinal.png "fig:") [fig:LSTM]

Convolutional Neural Network (CNN)
----------------------------------

As previously mentioned, also the CNN was tuned by including the disposition of the elements of the architecture as hyperparameters. In this case, both the number of convolutional blocks and filters, as well as the kernel, were taken into account. In the end, the structure resulted to be as Figure [fig:CNN] portrays, having the following setup:

-   Epochs: 200

-   Learning Rate: 0.005

-   Batch Size: 32

-   Kernel size: 7

![The final CNN resulting from the hyperparameters’ tuning](imgs/CNNFinal.png "fig:") [fig:CNN]

Random Forest Regression and Impurity 
--------------------------------------

The Random Forest Regression models allowed to evaluate further statistics on the explained variance of each future on the final prediction. This can be done by the model by monitoring how the OOB (out-of-the-bag) error evolves according to the splits at each node and the features excluded by each estimator. As Table [tab:explainedvar] summarizes, over 95% of the total variance is explained by 5 of the fastest moving averages, being another confirmation of the aforementioned Feature Engineering.

[!htbp]

[tab:explainedvar]

Error Measures: Pearson Correlation, Dynamic Time Warping (DTW) and Fast DTW
============================================================================

The first measure the author decided to introduce was a correlation term, which was able to correctly capture the general trend of the true time series. The Pearson Correlation ([eq:pearson]) thus represented an as simple as effective mathematical indicator for what concerns the spatial evolution of the financial trend.

\[\label{eq:pearson}
    \rho_{XY}= \frac{cov(X,Y)}{\sigma(x)\sigma(Y)}\]

In the deep learning experiments it was firstly used retrospectively as a method to favour those models which not only minimized the distance with respect to the real trend (i.e. low MSE), but also which were able to maximize the correlation. The Equation ([eq:acceptance]) models the decision rule adopted at the end of each training procedure:

\[\label{eq:acceptance}
    \underset{\theta}{\mathrm{argmin}}~ \mathrm{MSE}_{\hat{y},y} \cdot (1-\rho_{\hat{y},y})\]

This simple mathematical combination ensured to have a more global view on the outcomes and to increase the overall optimality.
The next tentative mainly dealt with the plug-in of such decision rule inside the loss function itself, but, due to the consequent increase of complexity and training times of the networks structure, this task was left for future improvements.
Given the aforementioned spatial modelling, the intent was then to include in the error measures an indicator which could symmetrically act on the temporal domain. Indeed, when dealing with random walk time series (as the financial case is), predictive models are distinguished by systematically missing the so-called *pivot* points, i.e. where the trend changes its concavity (on the local minima and maxima). This often results in a delay in the prediction of these points, since each model prediction would be obtained based on the high autocorrelation existing with the last known sample.
In literature ,, there are several studies about the most correct temporal measures/loss to introduce in deep learning models and in this project two of them have been adopted to verify whether an increase of the complexity is justified with respect to their effectiveness.
The first measure is the so-called *Dynamic Time Warping*. Behind the computation of the DTW there is a solution of a Quadratic Optimization Problem making use of Dynamic Programming. The value of the indicator is the value of the shortest path built between two time series, according to a *window* parameter, which regulates a one-to-one unidirectional mapping from the predicted output and the real trend.
Despite its effectiveness in describing the temporal nature of the predictions, its complexity (\(O(n^2)\)) is quite limiting, especially whether there is the necessity to plug it into a loss function. As did with the Pearson correlation coefficient, its evaluation was done retrospectively according to the following updated decision rule:

\[\underset{\theta}{\mathrm{argmin}}~ \mathrm{MSE}_{\hat{y},y} \cdot (1-\rho_{\hat{y},y})\cdot \mathrm{DTW}_{\hat{y},y, window}\]

As stated before, some brief experiments were held by customizing the loss function used in deep learning models, but that would have been too time consuming.
The last indicator reported for the sake of completeness is the Fast-DTW (*Fast Dynamic Time Warping*) . It is a simplified, but equally accurate version of the DTW, bounding the calculations to a linear complexity.

Details: Trading Strategy
=========================

The trading strategy is an event-based simulation, automatically deciding whether to enter or not the market for each time \(t\), based on the price evolution predicted at \(t+1\). In doing so, it is checked if the current predicted price \(\tilde{p}(t)\) represents a local minimum (i.e. \(\tilde{p}(t-1)>\tilde{p}(t)<\tilde{p}(t+1)\)) or maximum (i.e. \(\tilde{p}(t-1)<\tilde{p}(t)>\tilde{p}(t+1)\) ) to then open a BUY or a SELL position, respectively.
In opening an order, of course, the real price is considered as the entry price. A Take Profit (TP) threshold is set by considering the predicted delta. A Stop Loss (SL) value is determined too, based on the \(risk Factor\) parameter the strategy is configured. This parameter regulates the predisposition to risk of the investor: the higher the \(risk Factor\), the minor the risk, and thus the closer will be the SL with respect to the TP threshold. Also in these cases, both the TP and SL values are referred to true prices.
The trading method then defines a \(budget\) value as the initial capitalization and a \(margin Call\) value, simulating the notification a broker would take in case the investments became too lossy. In this case, the strategy automatically limits the possibility to invest large quantities.
These quantities are also governed by \(leverage\) parameter, which automatically allows to determine an upper bound on the lot measure, represented by the \(maximumLot\) parameter. This is continuously updated by taking into consideration the available margin (i.e. the free capitalization quota of the budget above the \(margin Call\)) and the adopted leverage. The so-limited \(lot\) measure to be invested will be then calculated according to the outcome of the previous transactions: profitable transactions will allow to increase it by a \(step Lot\) measure, while lossy investments will downgrade the previous \(lot\) value of two steps (i.e. in this sense, by penalizing more the prediction errors of the algorithms, this can be thought as another countermeasure towards risk).
Finally the trading strategy accepts a \(smooth Factor \in (0,1]\) parameter, which determines how much confidence the trader wants to give to the predictor: by choosing a low \(smooth Factor\) the trader would pursue a more conservative approach, by smoothing the predictions of the variations and thus limiting the decisions of the model.
After an order is opened and the available margin is decreased, at each subsequent time interval it is monitored by coherently increasing/decreasing the budget according to how the true price evolved; eventually, the order is closed if the Take Profit (respectively the Stop Loss) was met in the transition from the previous time frame. To do so, the High and the Low prices of the time frames are considered to check if the extreme variations of the price overcame one of the two thresholds.
Despite the very detailed implementation of this automatic trading strategy, the algorithm has to cope with the limits of the time granularity. It means that the algorithm is not able to determine if, in a time interval where the real price met both the TP and SL, it had hit before one than the other. In order to accomplish a worst case scenario (i.e. the minimum achievable profit), in such cases the algorithm always registers a loss. In general, this decision process happens quite often (about 30% of the total market orders is characterized by this ambiguity), therefore the results are still more pessimistic than the real materializable worst situation.

[!hbtp]

![image](imgs/RandomForest.png)

[fig:RandomForest]

[!hbtp]

![image](imgs/TCN.png)

[fig:TCNoutput]

[!hbtp]

![image](imgs/RNN.png)

[fig:LSTMoutput]

[!hbtp]

<span>@|c|ccccc|@</span> & & & & & **Fast DTW**
 & AUDCAD\_D1 & **2,282E-05** & 0,990 & 1,69 & 1,46
 & AUDCAD\_H1 & **4,171E-05** & 0,978 & 59,01 & 42,95
 & AUDCAD\_H4 & 6,919E-05 & 0,990 & 28,96 & 19,01
 & AUDCAD\_M30 & 3,070E-04 & 0,781 & 268,10 & 252,54
 & AUDUSD\_D1 & **4,130E-05** & 0,988 & 3,02 & 2,18
 & AUDUSD\_H1 & 4,724E-04 & 0,973 & 251,04 & 128,29
 & AUDUSD\_H4 & 8,136E-05 & 0,983 & 30,13 & 20,18
 & AUDUSD\_M30 & 2,540E-04 & 0,930 & 377,06 & 281,21
 & EURAUD\_D1 & 9,708E-05 & 0,981 & 3,83 & 3,22
 & EURAUD\_H1 & 1,103E-04 & 0,990 & 95,27 & 70,21
 & EURAUD\_H4 & 1,504E-04 & 0,970 & 36,57 & 25,94
 & EURAUD\_M30 & 1,221E-04 & 0,993 & 251,26 & 204,85
 & EURCAD\_D1 & 5,195E-05 & 0,984 & 2,46 & 2,25
 & EURCAD\_H1 & 8,463E-05 & 0,988 & 85,76 & 68,55
 & EURCAD\_H4 & 1,096E-04 & 0,982 & 33,96 & 25,05
 & EURCAD\_M30 & 2,868E-04 & 0,984 & 422,07 & 229,64
 & EURCHF\_D1 & **1,322E-05** & 0,996 & 1,52 & 1,26
 & EURCHF\_H1 & 7,315E-04 & 0,985 & 338,45 & 142,56
 & EURCHF\_H4 & 4,015E-04 & 0,998 & 87,52 & 36,08
 & EURCHF\_M30 & 5,886E-04 & 0,987 & 694,78 & 345,70
 & EURGBP\_D1 & **2,365E-05** & 0,978 & 2,03 & 1,76
 & EURGBP\_H1 & 1,162E-04 & 0,995 & 108,21 & 62,88
 & EURGBP\_H4 & 1,988E-04 & 0,963 & 56,79 & 30,39
 & EURGBP\_M30 & 4,716E-04 & 0,969 & 630,76 & 225,34
 & EURJPY\_D1 & 5,406E-01 & 0,991 & 338,90 & 275,40
 & EURJPY\_H1 & 1,005E+00 & 0,995 & 9520,17 & 7855,89
 & EURJPY\_H4 & 9,954E-01 & 0,994 & 3533,22 & 2173,67
 & EURJPY\_M30 & 5,621E-01 & 0,994 & 17096,88 & 15978,70
 & EURUSD\_D1 & **2,403E-05** & 0,992 & 1,56 & 1,46
 & EURUSD\_H1 & **3,350E-05** & 0,975 & 52,33 & 41,98
 & EURUSD\_H4 & 6,612E-05 & 0,986 & 25,01 & 20,31
 & EURUSD\_M30 & **3,846E-05** & 0,995 & 139,97 & 111,81
 & GBPUSD\_D1 & 5,015E-05 & 0,988 & 2,08 & 1,95
 & GBPUSD\_H1 & 1,423E-04 & 0,997 & 119,12 & 84,56
 & GBPUSD\_H4 & 1,283E-03 & 0,989 & 153,71 & 57,55
 & GBPUSD\_M30 & 1,821E-04 & 0,985 & 319,89 & 217,21
 & USDCAD\_D1 & **3,763E-05** & 0,982 & 1,85 & 1,67
 & USDCAD\_H1 & 3,657E-04 & 0,915 & 183,11 & 154,36
 & USDCAD\_H4 & 1,595E-04 & 0,958 & 45,22 & 36,86
 & USDCAD\_M30 & 1,744E-04 & 0,936 & 325,22 & 249,26
 & USDJPY\_D1 & 2,150E-01 & 0,981 & 151,00 & 139,61
 & USDJPY\_H1 & 4,139E-01 & 0,997 & 5735,21 & 4389,00
 & USDJPY\_H4 & 1,192E-01 & 0,990 & 907,00 & 792,73
 & USDJPY\_M30 & 3,071E-01 & 0,995 & 12732,86 & 9150,70

[tab:TCN]

[!htbp]

<span>@|c|ccccc|@</span> & & & & & **Fast DTW**
 & AUDCAD\_D1 & **2,195E-05** & 0,990 & 1,45 & 1,28
 & AUDCAD\_H1 & **3,617E-05** & 0,981 & 55,20 & 41,24
 & AUDCAD\_H4 & 5,838E-05 & 0,989 & 25,34 & 17,55
 & AUDCAD\_M30 & **2,355E-05** & 0,982 & 104,40 & 82,91
 & AUDUSD\_D1 & **1,961E-05** & 0,993 & 1,41 & 1,31
 & AUDUSD\_H1 & 3,196E-04 & 0,983 & 213,32 & 90,60
 & AUDUSD\_H4 & 1,344E-04 & 0,991 & 43,41 & 21,35
 & AUDUSD\_M30 & 3,397E-04 & 0,983 & 527,02 & 186,37
 & EURAUD\_D1 & 6,799E-05 & 0,986 & 1,74 & 1,69
 & EURAUD\_H1 & 5,010E-05 & 0,992 & 58,13 & 49,73
 & EURAUD\_H4 & 5,463E-05 & 0,991 & 19,29 & 17,04
 & EURAUD\_M30 & 5,515E-05 & 0,996 & 162,97 & 132,90
 & EURCAD\_D1 & **4,620E-05** & 0,984 & 1,62 & 1,56
 & EURCAD\_H1 & 6,444E-05 & 0,991 & 71,59 & 56,98
 & EURCAD\_H4 & 1,216E-04 & 0,989 & 39,24 & 26,09
 & EURCAD\_M30 & 2,007E-04 & 0,992 & 363,27 & 177,56
 & EURCHF\_D1 & **1,192E-05** & 0,997 & 1,20 & 1,11
 & EURCHF\_H1 & 3,450E-04 & 0,991 & 231,61 & 77,20
 & EURCHF\_H4 & 2,783E-04 & 0,998 & 72,47 & 28,74
 & EURCHF\_M30 & 2,137E-04 & 0,997 & 420,91 & 186,04
 & EURGBP\_D1 & **1,834E-05** & 0,977 & 1,11 & 1,10
 & EURGBP\_H1 & 1,149E-04 & 0,996 & 114,22 & 54,69
 & EURGBP\_H4 & 1,191E-04 & 0,979 & 43,35 & 20,76
 & EURGBP\_M30 & 3,562E-04 & 0,982 & 548,42 & 190,59
 & EURJPY\_D1 & 8,072E+00 & 0,987 & 2222,41 & 1151,29
 & EURJPY\_H1 & 1,032E+01 & 0,992 & 38094,83 & 16649,40
 & EURJPY\_H4 & 8,810E+00 & 0,984 & 12406,93 & 5253,50
 & EURJPY\_M30 & 9,755E+00 & 0,988 & 87681,32 & 41160,72
 & EURUSD\_D1 & **2,081E-05** & 0,993 & 0,94 & 0,91
 & EURUSD\_H1 & 6,735E-05 & 0,985 & 84,78 & 49,01
 & EURUSD\_H4 & **2,301E-05** & 0,997 & 14,65 & 10,45
 & EURUSD\_M30 & 1,020E-04 & 0,998 & 273,31 & 133,33
 & GBPUSD\_D1 & 5,496E-05 & 0,987 & 1,90 & 1,73
 & GBPUSD\_H1 & **4,656E-05** & 0,998 & 62,17 & 51,78
 & GBPUSD\_H4 & 1,149E-03 & 0,992 & 146,28 & 57,16
 & GBPUSD\_M30 & **3,726E-05** & 0,993 & 138,76 & 114,91
 & USDCAD\_D1 & **3,633E-05** & 0,982 & 1,59 & 1,48
 & USDCAD\_H1 & 3,807E-04 & 0,988 & 224,63 & 100,90
 & USDCAD\_H4 & 9,538E-05 & 0,982 & 34,00 & 21,88
 & USDCAD\_M30 & 1,401E-04 & 0,983 & 302,48 & 177,57
 & USDJPY\_D1 & 7,112E-01 & 0,953 & 452,03 & 356,08
 & USDJPY\_H1 & 5,054E+00 & 0,995 & 24448,67 & 13890,73
 & USDJPY\_H4 & 7,275E-01 & 0,953 & 2842,60 & 1998,01
 & USDJPY\_M30 & 2,626E+00 & 0,981 & 41922,36 & 19898,85

[tab:RNN]

[!hbtp]

<span>@|c|ccccc|@</span> & & & & & **Fast DTW**
 & AUDCAD\_D1 & 6,050E-05 & 0,974 & 4,00 & 2,40
 & AUDCAD\_H1 & **4,191E-05** & 0,966 & 58,87 & 43,34
 & AUDCAD\_H4 & 5,876E-05 & 0,980 & 24,56 & 15,12
 & AUDCAD\_M30 & 6,355E-05 & 0,960 & 176,03 & 128,90
 & AUDUSD\_D1 & 1,132E-04 & 0,968 & 4,67 & 2,72
 & AUDUSD\_H1 & 3,561E-04 & 0,950 & 211,64 & 103,54
 & AUDUSD\_H4 & 4,497E-04 & 0,923 & 69,00 & 32,99
 & AUDUSD\_M30 & 4,754E-04 & 0,947 & 602,55 & 241,68
 & EURAUD\_D1 & 8,376E-04 & 0,983 & 19,78 & 9,72
 & EURAUD\_H1 & 2,537E-04 & 0,975 & 151,61 & 80,69
 & EURAUD\_H4 & 6,203E-04 & 0,978 & 86,80 & 38,42
 & EURAUD\_M30 & 2,368E-04 & 0,989 & 366,31 & 200,38
 & EURCAD\_D1 & 8,827E-05 & 0,982 & 4,54 & 3,08
 & EURCAD\_H1 & 1,589E-04 & 0,981 & 121,01 & 87,25
 & EURCAD\_H4 & 7,998E-05 & 0,982 & 26,49 & 19,51
 & EURCAD\_M30 & 1,009E-04 & 0,983 & 239,12 & 148,18
 & EURCHF\_D1 & 2,327E-04 & 0,929 & 10,62 & 6,05
 & EURCHF\_H1 & 3,879E-04 & nan & 192,23 & 192,23
 & EURCHF\_H4 & 1,894E-04 & 0,940 & 52,34 & 30,44
 & EURCHF\_M30 & 5,446E-04 & 0,917 & 596,38 & 299,90
 & EURGBP\_D1 & **4,288E-05** & 0,972 & 3,42 & 2,44
 & EURGBP\_H1 & 2,881E-04 & 0,977 & 181,29 & 119,98
 & EURGBP\_H4 & 4,736E-04 & 0,954 & 90,72 & 40,84
 & EURGBP\_M30 & 5,055E-04 & 0,937 & 633,99 & 239,59
 & EURJPY\_D1 & 2,166E+01 & 0,965 & 3682,40 & 1667,98
 & EURJPY\_H1 & 2,943E+01 & 0,990 & 63962,81 & 24509,04
 & EURJPY\_H4 & 2,062E+01 & 0,985 & 19279,09 & 6226,13
 & EURJPY\_M30 & 3,097E+01 & 0,988 & 157722,62 & 71165,37
 & EURUSD\_D1 & 1,139E-04 & 0,961 & 6,02 & 5,53
 & EURUSD\_H1 & 1,004E-04 & 0,938 & 99,15 & 59,78
 & EURUSD\_H4 & 2,190E-04 & 0,964 & 54,67 & 29,37
 & EURUSD\_M30 & 2,306E-04 & 0,960 & 363,82 & 202,83
 & GBPUSD\_D1 & 3,909E-04 & 0,906 & 11,41 & 7,82
 & GBPUSD\_H1 & 8,662E-04 & 0,987 & 325,71 & 141,28
 & GBPUSD\_H4 & 7,337E-04 & 0,916 & 88,12 & 55,31
 & GBPUSD\_M30 & 6,687E-04 & 0,905 & 612,46 & 340,30
 & USDCAD\_D1 & 1,805E-04 & 0,955 & 7,26 & 4,68
 & USDCAD\_H1 & 4,763E-04 & 0,964 & 217,13 & 99,00
 & USDCAD\_H4 & 1,886E-04 & 0,964 & 43,81 & 22,86
 & USDCAD\_M30 & 3,696E-04 & 0,934 & 459,36 & 191,22
 & USDJPY\_D1 & 1,153E+02 & 0,872 & 8908,74 & 8908,74
 & USDJPY\_H1 & 1,931E+02 & 0,995 & 170016,25 & 139862,46
 & USDJPY\_H4 & 1,181E+02 & 0,952 & 47803,52 & 47803,52
 & USDJPY\_M30 & 1,422E+02 & 0,987 & 351743,18 & 340365,57

[tab:CNN]

[!htbp]

<span>@|ccccc|@</span> **Currency** & **MSE** & **Pearson Correlation** & **DTW** & **Fast DTW**
EURGBP & **1,524E-05** & 0,981 & 1,18 & 1,15
USDJPY & 1,969E-01 & 0,981 & 105,38 & 102,58
AUDUSD & **1,704E-05** & 0,993 & 1,25 & 1,20
AUDCAD & **1,607E-05** & 0,993 & 1,42 & 1,37
EURUSD & **2,058E-05** & 0,993 & 1,13 & 1,13
USDCAD & **2,664E-05** & 0,987 & 1,40 & 1,34
EURCAD & **3,814E-05** & 0,987 & 2,06 & 1,95
GBPUSD & **4,978E-05** & 0,988 & 1,88 & 1,89
EURCHF & <span> **9,702E-06**</span> & 0,997 & 1,03 & 0,98
EURJPY & 3,209E-01 & 0,992 & 146,28 & 139,88
EURAUD & **6,046E-05** & 0,987 & 2,01 & 1,92

[tab:LinearRegression]

[!htbp]

<span>@|ccccccc|@</span> & & & & & & **Tolerance**
EURGBP & **1,557E-05** & 0,980 & 0,60 & 0,61 & 100 & 1,00E-10
USDJPY & 1,996E-01 & 0,981 & 79,71 & 77,96 & 100 & 1,00E-06
AUDUSD & **1,601E-05** & 0,994 & 1,03 & 1,06 & 100 & 1,00E-06
AUDCAD & **1,663E-05** & 0,993 & 0,82 & 0,80 & 100 & 1,00E-10
EURUSD & **2,006E-05** & 0,993 & 0,90 & 0,89 & 100 & 1,00E-10
USDCAD & **2,890E-05** & 0,985 & 1,03 & 1,00 & 100 & 1,00E-10
EURCAD & **3,990E-05** & 0,986 & 1,08 & 1,05 & 100 & 1,00E-10
GBPUSD & **4,934E-05** & 0,988 & 1,52 & 1,50 & 100 & 1,00E-10
EURCHF & <span> **8,624E-06**</span> & 0,997 & 0,67 & 0,64 & 100 & 1,00E-06
EURJPY & 3,053E-01 & 0,992 & 112,23 & 111,61 & 100 & 1,00E-06
EURAUD & **6,426E-05** & 0,986 & 1,39 & 1,36 & 100 & 1,00E-10

[tab:BayesianRidgeRegression]

[!hbtp]

<span>@|ccccccccc|@</span> & & & & & & & & **Gamma**
EURGBP & 4,253E-04 & 0,862 & 14,61 & 7,13 & Poly & 3 & 10 & 0,001
USDJPY & 1,949E-01 & 0,981 & 66,82 & 65,82 & Linear & 1 & 10 & 0,0001
AUDUSD & 4,018E-04 & 0,968 & 15,33 & 5,69 & Poly & 1 & 10 & 0,0001
AUDCAD & 6,292E-04 & 0,956 & 16,98 & 11,94 & RBF & 1 & 1 & 0,0001
EURUSD & 1,360E-03 & 0,972 & 27,50 & 15,02 & Poly & 2 & 10 & 0,001
USDCAD & 9,294E-04 & 0,919 & 22,45 & 10,85 & Poly & 2 & 1 & 0,001
EURCAD & 3,687E-04 & 0,917 & 11,03 & 5,52 & Poly & 2 & 1 & 0,01
GBPUSD & 2,341E-04 & 0,943 & 8,26 & 4,49 & Poly & 1 & 1 & 0,01
EURCHF & 3,658E-04 & 0,989 & 14,97 & 6,78 & RBF & 1 & 1 & 0,0001
EURJPY & 2,968E-01 & 0,993 & 85,89 & 84,83 & Linear & 1 & 10 & 0,0001
EURAUD & 1,123E-03 & 0,867 & 20,41 & 14,01 & Poly & 2 & 10 & 0,001

[tab:SupportVector]

[!hbtp]

<span>@|cccccc|@</span> & & & & & **Number of Estimators**
EURGBP & <span> **5,225E-06**</span> & 0,993 & 0,99 & 0,98 & 500
USDJPY & 6,108E-02 & 0,994 & 111,63 & 110,21 & 500
AUDUSD & <span> **5,477E-06**</span> & 0,998 & 1,07 & 1,06 & 500
AUDCAD & <span> **5,946E-06**</span> & 0,997 & 1,03 & 1,02 & 500
EURUSD & <span> **7,338E-06**</span> & 0,997 & 1,18 & 1,17 & 500
USDCAD & **1,064E-05** & 0,995 & 1,43 & 1,41 & 500
EURCAD & **1,540E-05** & 0,995 & 1,65 & 1,62 & 500
GBPUSD & **1,808E-05** & 0,996 & 1,80 & 1,78 & 500
EURCHF & <span> **2,730E-06**</span> & 0,999 & 0,78 & 0,76 & 500
EURJPY & 1,004E-01 & 0,998 & 139,53 & 138,21 & 500
EURAUD & **2,147E-05** & 0,995 & 1,92 & 1,91 & 500

[tab:RandomForest]

[!htbp]

<span>@|ccccccc|@</span> & & & & & & **Internal Algorithm**
EURGBP & **1,138E-05** & 0,986 & 0,72 & 0,71 & 50 & Ball Tree
USDJPY & 1,387E-01 & 0,987 & 118,17 & 111,42 & 50 & Brute
AUDUSD & **3,190E-05** & 0,989 & 1,41 & 1,37 & 50 & Brute
AUDCAD & **1,050E-05** & 0,995 & 0,98 & 0,93 & 50 & Brute
EURUSD & **1,587E-05** & 0,995 & 0,92 & 0,91 & 50 & Ball Tree
USDCAD & **2,815E-05** & 0,986 & 1,08 & 1,04 & 50 & Ball Tree
EURCAD & **2,736E-05** & 0,991 & 0,98 & 0,95 & 50 & Ball Tree
GBPUSD & **4,747E-05** & 0,989 & 1,58 & 1,59 & 50 & Ball Tree
EURCHF & **1,587E-05** & 0,994 & 0,92 & 0,94 & 50 & Ball Tree
EURJPY & 2,185E-01 & 0,995 & 111,94 & 114,07 & 50 & Ball Tree
EURAUD & 9,962E-05 & 0,978 & 1,72 & 1,71 & 50 & Ball Tree

[tab:KNN]

[

]

[1] Data retrieved in April 2019 by the Bank for International Settlement

[2] https://eaforexacademy.com/software/forex-historical-data/


# stock2vec

This project is to transform stock data into vectors, and run machine learning modal to find related context among stocks.

There are two models experimented:
*  CBOW
   
   Input data are target ticker and the list of top most correlated tickers for the target ticker represented its context,

   The model produces links by predicting the correlation between context and target ticker
*  Skip_Gram(output more accurate)
   
   Input data are target ticker and the list of top most correlated tickers,

   The model analyzes frequency of correlated tickers occurred around target ticker, and creates vectors based on the frequency

<img src="https://github.com/jojonki/word2vec-pytorch/blob/master/word2vec.PNG?raw=true" >

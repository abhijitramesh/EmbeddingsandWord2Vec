# Word Embedding
We know that Computers are good with numbers which by association means Neural Networks are good with numbers but not so great with text. In order to do text representation what we usually do in called one-hot encoding where we take a vocabulary and represent it as a vector. Work Embedding means creating a neural network which can learn from a vocabulary of words to represent words as vectors but the catch here is this can also learn some more interesting things like association between words like tenses for example complete and completed. They can also learn gender bases associations like woman and queen or man and king.


# Embedded Weights/Lookup Table
If we look at a Rnn which uses text input probably they have a one-hot encoded vector which is about thousands of columns in size but has only one value set to one. Then this matrix is multiplied by the weight matrix but since most of the value in it is zero what we want for is multiplied on the one corresponding to the set (the value in vector which is one) value. So what we can use here in order to avoid such waste of computational resources is using an embedding layer which is then again a hidden layer which is basically a matrix to which the one-hot encoded value is multiplied initially, the result would be only the value in this new layer which represents a unique number corresponding to a words and hence this can be used as a look up table if we want to know what text this numerical value is actually representing. The perk here is that the new rest of the calculation can be done using this numerical data instead of using a huge memory hungry vector.

[Click here](https://towardsdatascience.com/what-the-heck-is-word-embedding-b30f67f01c81) for an interesting article in Word Embedding.

# Skip gram Work2Vec
As we have said earlier it is very compute intensive to use the one-hot encoded vector at every step so what we can do is instead use an embedding layer to skip the unnecessary multiplication step.

[Click here](https://github.com/abhijitramesh/EmbeddingsandWord2Vec/blob/master/Skip_Grams.ipynb) to see the implementation of Skip Gram Word to vec

#### [Utils.py](https://github.com/abhijitramesh/EmbeddingsandWord2Vec/blob/master/utils.py)
This file contains two functions:
 ```python
def preprocess(text):
```
This function is used to mainly to tokenize the data initially everything is made to small letters and the symbols are replaced with appropriate tokens. Then a counter is initialized to get the frequency of words and if the words occurs less than 5 times it is discarded this improves the performance of the neural net.

```python
def create_lookup_tables(words):
```

This function returns two dictionaries one which maps the words to integers and the other which maps the integer to words.

## Subsampling
There might be some noise in the data words like the/or may not provide much context about the surrounding words and hence we can remove this by a process called subsampling. The formula representing subsampling is given by:

<img src="https://render.githubusercontent.com/render/math?math=P(W_i) = 1 - \sqrt{\frac{t}{f(w_i)}}">

Where t is the threshold we set and w_i is the words for which we are calculating the probability if it should be removed or not. f(wi) is the frequency that the words appears in the dataset.
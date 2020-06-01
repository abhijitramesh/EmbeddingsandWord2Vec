# Word Embedding
We know that Computers are good with numbers which by association means Neural Networks are good with numbers but not so great with text. In order to do text representation what we usually do in called one-hot encoding where we take a vocabulary and represent it as a vector. Work Embedding means creating a neural network which can learn from a vocabulary of words to represent words as vectors but the catch here is this can also learn some more interesting things like association between words like tenses for example complete and completed. They can also learn gender bases associations like woman and queen or man and king.


# Embedded Weights/Lookup Table
If we look at a Rnn which uses text input probably they have a one-hot encoded vector which is about thousands of columns in size but has only one value set to one. Then this matrix is multiplied by the weight matrix but since most of the value in it is zero what we want for is multiplied on the one corresponding to the set (the value in vector which is one) value. So what we can use here in order to avoid such waste of computational resources is using an embedding layer which is then again a hidden layer which is basically a matrix to which the one-hot encoded value is multiplied initially, the result would be only the value in this new layer which represents a unique number corresponding to a words and hence this can be used as a look up table if we want to know what text this numerical value is actually representing. The perk here is that the new rest of the calculation can be done using this numerical data instead of using a huge memory hungry vector.

[Click here](https://towardsdatascience.com/what-the-heck-is-word-embedding-b30f67f01c81) for an interesting article in Word Embedding.

# Skip gram Work2Vec
As we have said earlier it is very compute intensive to use the one-hot encoded vector at every step so what we can do is instead use an embedding layer to skip the unnecessary multiplication step.

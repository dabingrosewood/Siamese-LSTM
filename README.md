Siamese-LSTM RNN for semantic sentence search
==== 

This project is actually for the IRTA course in leiden Uni.



Target:
    to detect the semantic duplicate from pairs of sentence.
    
用途：用来记录语意相似文本


the basic idea behind this program is [this paper](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf).

####environment环境:
    python 3.6.4
    tensorflow 1.8
    nltk 3.2.5
    keras 2.1.5
    sklearn 0.19
    gensim 1.7


###reference:
    https://github.com/likejazz/Siamese-LSTM
    https://github.com/aditya1503/Siamese-LSTM
    https://blog.csdn.net/android_ruben/article/details/78427068

####note:

    the pre-trained word vector has been used(numberbatch-17-06_en in the test).both word2vec and numberbatch are recommended


#####result:
    acc on training data: 0.8670
    acc on test data: 0.8255

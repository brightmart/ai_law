AI_LAW
-------------------------------------------------------------------------
all kinds of baseline models for smart law use AI.


1.Desc
-------------------------------------------------------------------------
this repository contain models that learn from law cases to predict:

   crime(accusation): which kinds of crime the bad guy did according to law system

   relevant articles: specific law that is used for this case

   term of imprisonment: whether it is death penalty or life imprisonment, how many years in prison(imprisonment)

find more about task, data or even start smart AI competition by check here:

 <a href='http://cail.cipsc.org.cn/index.html'>http://cail.cipsc.org.cn/index.html</a>



2.Data Visualization

 1)total examples, crime,relevant articles:

  data_train: 155k

  data_valid: 17k

  data_test: 33k

  number of total [accusation]: 202

  number of total [relevant_article]:183

  length of inputs(facts of the law case, after word tokenize)

     average length: 279

     max length: 9

     mini length: 19447

     length percentage:[(50, 0.01), (75, 0.04), (100, 0.07), (150, 0.19), (200, 0.19), (300, 0.25), (400, 0.11), (500, 0.05), (600, 0.03), (700, 0.06)]

     as we can see the majority of length is between 100 and 500. percentage of length less than 400 is 87%, less then 500 is  0.92.

     notice: here (50, 0.01) means length less than 50 is 1%, (75, 0.04) means length greater than 50 and less than 75  is 4%.

     that's take length as 500. pad for length less it, and truncate words exceed this number(truncate from start words, not end).

  label distribution(accusation, relevant article):

     how many accusations for a case?

         average length:1.2; max_length:6;mini_length:1; around 78% cases only has one accusation. most cases's accusation is less and equal than 3.

         length of accusation distribution:[(1, 0.78), (2, 0.0), (3, 0.2), (4, 0.019), (5, 0.002), (6, 0.001), (7, 0.0001)]

     how many relevant articles for a case?

         average length:1.45; max_length:9; mini_length:1; around 70% cases only has one relevant articles;

         length of relevant articles:[(1, 0.71), (2, 0.0), (3, 0.18), (4, 0.07), (5, 0.03), (6, 0.01), (7, 0.004)].


3.Feature Engineering TODO
-------------------------------------------------------------------------



4.Imbalance Classification for Skew Data TODO
-------------------------------------------------------------------------


5.Transfer Learning & Pretrained Word Embedding TODO
-------------------------------------------------------------------------




6.Models TODO
-------------------------------------------------------------------------
1) fastText

2) TextCNN

3) HAN: hierarchical attention network



7.Performance
-------------------------------------------------------------------------
   performance on validation dataset(seperate from training data):

Task |Model | F1 Score
--|--|--|
fastText| |
TextCNN| |
HAN||



8.Error Analysis
----------------------------------------------------------------




9.Usage
-------------------------------------------------------------------------
  1) transform your data:

    call transform_format from data_util_transform_to_standard_format.py

  2) train:

     python p7_TextCNN_train.py

    it will report macro f1 score and micro f1 score when doing validation.




10.Environment
-------------------------------------------------------------------------
   python 2.7 + tensorflow 1.8

   for people use python3, just comment out three lines below in the begining of file:

      import sys

      reload(sys)

      sys.setdefaultencoding('utf-8')


11.Model Details
-------------------------------------------------------------------------



12.TODO
-------------------------------------------------------------------------

   1) extract more data mining features

   2) use traditional machin learning like xgboost,random forest

   3) try some classic text classification network


13.Conclusion
-------------------------------------------------------------------------



14.Reference
-------------------------------------------------------------------------
  1) <a href='https://arxiv.org/pdf/1408.5882v2.pdf'>TextCNN:Convolutional Neural Networks for Sentence Classification</a>

  2) A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification

  7) <a href='https://github.com/facebookresearch/fastText'>fastText:Bag of Tricks for Efficient Text Classification</a>



if you are smart or can contribute new ideas, join with us.

to be continued. for any problem, contact brightmart@hotmail.com

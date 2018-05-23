AI_LAW
-------------------------------------------------------------------------
all kinds of baseline models for smart law use AI.

Update: Joint Model for law cases prediction is released. run python HAN_train.py to train the model for predict accusation, relevant articles and term of imprisonment.

challenge of this task: the description of law case(called facts) is quite long, the average words is around 400. it not only long, but contain lots of information,

even for human, you need to pay lots of attention before you can tell what's it about. for machine it need to have ability to handle long distance dependence,

and focus on most important information.


if you have any suggestion or problem, a better idea on how to slove this problem,find a bug or want to find a team member, you are welcomed to can contact with me:

brightmart@hotmail.com  or commit your code to this repository.


1.Desc
-------------------------------------------------------------------------
this repository contain models that learn from law cases to predict:

   crime(accusation): which kinds of crime the bad guy did according to law system

   relevant articles: specific law that is used for this case

   term of imprisonment: whether it is death penalty or life imprisonment, how many years in prison(imprisonment)

find more about task, data or even start smart AI competition by check here:

<a href='https://github.com/brightmart/ai_law/blob/master/data/model_law.jpg'>https://github.com/brightmart/ai_law/blob/master/data/model_law.jpg</a>

<a href='http://cail.cipsc.org.cn/index.html'>http://cail.cipsc.org.cn/index.html</a>



2.Data Visualization & Pre-processing
-------------------------------------------------------------------------

 1) total examples, crime,relevant articles:

      data_train: 155k

      data_valid: 17k

      data_test: 33k

      number of total [accusation]: 202

      number of total [relevant_article]:183


  2) length of inputs(facts of the law case, after word tokenize)

     average length: 279

     max length: 9

     mini length: 19447

     length percentage:[(50, 0.01), (75, 0.04), (100, 0.07), (150, 0.19), (200, 0.19), (300, 0.25), (400, 0.11), (500, 0.05), (600, 0.03), (700, 0.06)]

     as we can see the majority of length is between 100 and 500. percentage of length less than 400 is 87%, less then 500 is  0.92.

     notice: here (50, 0.01) means length less than 50 is 1%, (75, 0.04) means length greater than 50 and less than 75  is 4%.

     that's take length as 500. pad for length less it, and truncate words exceed this number(truncate from start words, not end).


  3) label distribution(accusation, relevant article):

     how many accusations for a case?

         average length:1.2; max_length:6;mini_length:1; around 78% cases only has one accusation. most cases's accusation is less and equal than 3.

         length of accusation distribution:[(1, 0.78), (2, 0.0), (3, 0.2), (4, 0.019), (5, 0.002), (6, 0.001), (7, 0.0001)]

     how many relevant articles for a case?

         average length:1.45; max_length:9; mini_length:1; around 70% cases only has one relevant articles;

         length of relevant articles:[(1, 0.71), (2, 0.0), (3, 0.18), (4, 0.07), (5, 0.03), (6, 0.01), (7, 0.004)].



  4) Top20 accusation and its frequency( as you can see that data imbalance problem is not a big problem here.)

          盗窃:10051

          走私、贩卖、运输、制造毒品:8872

          故意伤害:6377

          抢劫:5020

          诈骗:3536

          受贿:3496

          寻衅滋事:3290

          危险驾驶:2758

          组织、强迫、引诱、容留、介绍卖淫:2647

          制造、贩卖、传播淫秽物品:2617

          容留他人吸毒:2597

          交通肇事:2562

          贪污:2391

          非法持有、私藏枪支、弹药:2349

          故意杀人:2282

          开设赌场:2259

          非法持有毒品:2203

          职务侵占:2197

          强奸:2192

          伪造、变造、买卖国家机关公文、证件、印章:2153

  5) preprocess value for imprisonment.

     range of imprisonment from 0 to 300(=12*25), it is raw value too big, not may lead to less efficient for model to learn.

     we will normalize imprisonment using following format:

          imprisonment= (imprisonment- imprisonment_mean)/imprisonment_std

     where imprisonment_mean stand for mean of imprisonment, imprisonment_std stand for stand deviation of imprisonment.

     we can easily to get its mean:, 26.2, and std:33.5 from training data. during test we will re-scale value back:

         imprisonment_test=(imprisonment_test+imprisonment_mean)*imprisonment_std



3.Evaluation: F1 score(Micro,Macro)
-------------------------------------------------------------------------
 for task1(predict accusation) and task2(predict relevant article), there are multi-label classification problem. and as we already seen from

 previous description, label distribution may be skewed or imbalanced. so we use f1 score which computed from precision and recall.

 rembember that:

       True Postive=TP=[predict=1,truth=1]

       False Postive=FP=[predict=1,truth=0]

       False Negative=FN=[predict=0,truth=1]

       precison=TP/(TP+FP). precision is among all labels that you predict as postive, how many are real postive

       recall=TP/(TP+FN).   recall is among all true labels, how many you predict as postive.

       f1=(2*precision*recall)/(precision+recall). it's value is among:[0,1].

![alt text](https://github.com/brightmart/ai_law/blob/master/data/f1_micro_macro.jpg)

       finally we compute:

            score=((f1_macro+f1_micro)/2.0)*100

  Question1: here we have multi-label, it is not binary classifcaiton problem, how do we define TP,FP,FN and f1 score?

       we will first get confusing matrix(TP,FP,FN), then compute f1 score.

       for your better understanding, i will give you an example. suppose target_label=[2,12,88], predict_label=[2,12,13,10]

       as you can see, unique_label=[2,12,13,10,88]. and two labels exists in both side, label_in_common=[2,12]. and we use a dict to store TP,FP,FN for each class.

       dict_count[class]=(num_TP,num_FP,num_FN)

       1) we will go though this unique_label, and count TP,FP,FN for each class:

            for the first element 2, it exists in predict_label(you predict it is true), and also exists in target_label(actually it is true),

                 so it is True Positive(TP), ===> dict[2]=(1,0,0)

            for the second element 12, it is similiar like first element, also True Positive(TP). ===> dict[12]=(1,0,0)

            for the third element 13, it not exists in predict_label(predict=0), but exists in target_label(truth=1),

                 so it is False Negative(FN),===> dict[13]=(0,0,1)

            for the fourth element 10, predict=1,truth=0 ===>False Postive(FP) ===> dict[10]=(0,1,0)

            for the last element 88,   predict=0,truth=1 ===>False Negative(FN) ===>dict[88]=(0,0,0)

       2) secondly, we compute total TP,FP,FN:

          TP=2,FP=1,FN=2

       3) finally, we compute P,R and F1 score:

          P_micro=TP/(TP+FP)=2/3=0.6777

          R_micro=TP/(TP+FN)=2/4=0.5

          F1_micro=(2*P_micro*R_micro)/(P_micro+R_micro)=0.575

       for detail, check compute_confuse_matrix() and compute_micro_macro() at evaluation_matrix.py


   Question2: the above steps is for only one input. but suppose after we go through several inputs, and got:

          dict_count[2]=(20,5,7)

          where we define format in above: dict_count[class]=(num_TP,num_FP,num_FN)

          what's the f1 score for class 2?

          this is similiar we saw in the last step(#3), we will compute Precsion,Recall, then f1 score:

             P_label2=TP/(TP+FP)=20/(20+5)=0.80

             R_label2=TP/(TP+FN)=20/(20+7)=0.74

             f1_label2=(2*P*R)/(P+R)=0.76

           notice: if you want to compute f1_macro, you can go through each class just same as label2, then compute average value among all classes as f1_macro.


  Question3: how many labels should we retrieve once we compute the logits for a input?

          remember we are in mulit-label classification scenario. given a input, there may exist multi-label at same time. we can't just get the max possible

          label as our target, which is usually implement by softmax function. that is saying that we can think this is a multi-binary classification problem,

          for each class, we just consider whether it is exists or not. by applying sigmoid function for each class, we will make sure that possibility will be

          between 0 and 1. if the possibility is greater(or equal) than 0.5, we think it is exists(predict=true).

          in a word, we look each class seperately, and get all classes that possibilities we predict greater than a threshold.


     for detail above evaluation matrix, check evaluation_matrix.py


4.Imbalance Classification for Skew Data TODO
-------------------------------------------------------------------------


5.Transfer Learning & Pretrained Word Embedding TODO
-------------------------------------------------------------------------
  download pretrained word embedding from https://github.com/Embedding/Chinese-Word-Vectors and enable flag 'use_embedding' during training.



6.Models TODO
-------------------------------------------------------------------------
1) fastText

2) TextCNN

3) HAN: hierarchical attention network(completed)



7.Performance
-------------------------------------------------------------------------
   performance on validation dataset(seperate from training data) train for first 1000 steps(around one epoch):

   ('1.Accasation Score:', 70.8571253641409, ';2.Article Score:', 71.28389708554336, ';3.Penalty Score:', 69.78947368421059, ';Score ALL:', 211.93049613389485)

![alt text](https://github.com/brightmart/ai_law/blob/master/data/han_1000steps.jpg)


Task |Model | F1 Score
--|--|--|
fastText| |
TextCNN| |
HAN||



8.Error Analysis
----------------------------------------------------------------




9.Usage
-------------------------------------------------------------------------
  train:

     python HAN_train.py

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
Hierarchical Attention Network:

Implementation of Hierarchical Attention Networks for Document Classification

Structure:

embedding

Word Encoder: word level bi-directional GRU to get rich representation of words

Word Attention:word level attention to get important information in a sentence

Sentence Encoder: sentence level bi-directional GRU to get rich representation of sentences

Sentence Attention: sentence level attention to get important sentence among sentences

One Layer MLP for transform document representation to each sub task's features.

FC+Softmax

![alt text](https://github.com/brightmart/ai_law/blob/master/data/model_law.jpg)

![alt text](https://github.com/brightmart/text_classification/blob/master/images/HAN.JPG)


Chinese Desc of Task:
-------------------------------------------------------------------------
“中国法研杯”司法人工智能挑战赛

挑战赛任务：

任务一  罪名预测：根据刑事法律文书中的案情描述和事实部分，预测被告人被判的罪名；

任务二  法条推荐：根据刑事法律文书中的案情描述和事实部分，预测本案涉及的相关法条；

任务三  刑期预测：根据刑事法律文书中的案情描述和事实部分，预测被告人的刑期长短。

数据集共包括268万刑法法律文书，共涉及183条罪名，202条法条，刑期长短包括0-25年、无期、死刑.


12.TODO
-------------------------------------------------------------------------

   1) tracking miss match of micro and macro of f1 score: balance micro and macro to maximize final f1 score

   2) error analysis: print and analysis error cases for each task, and get insight for improvement

   3) truncate or pad sequences in the beginning, or reverse

   4) preprocess document as serveral sentences before graph model

   5) try pure CNN or attention models to speed up training


13.Conclusion
-------------------------------------------------------------------------

  1) it is possible to solve the problem in a way of joint model. each sub task shared same input and representation.

  2) a single evaluation matrix(here micro and macro of f1 score) is important for evaluate how well you done on your task,

  and to know how well it is when you changed something.

  3) add more here.



14.Reference
-------------------------------------------------------------------------
  1) <a href='https://arxiv.org/pdf/1408.5882v2.pdf'>TextCNN:Convolutional Neural Networks for Sentence Classification</a>

  2) A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification

  3) <a href='https://github.com/facebookresearch/fastText'>fastText:Bag of Tricks for Efficient Text Classification</a>

  4) Hierarchical Attention Networks for Document Classification



if you are smart or can contribute new ideas, join with us.

to be continued. for any problem, contact brightmart@hotmail.com

AI_LAW
-------------------------------------------------------------------------
all kinds of baseline models for long text classificaiton( text categorization)


Update: Joint Model for law cases prediction is released.

run python HAN_train.py to train the model for predict accusation, relevant articles and term of imprisonment.

challenges of this task:

   1) the description of law case(called facts) is quite long, the average words is around 400. it is not only long, but contain lots of information,

      even for a human being, you need to pay lots of attention before you can tell what's it about. for machine it need to have ability to handle long distance dependency,

      and pay attention on most important information.

   2) multiple sub tasks are included in this task. you are not only to predict accusations, but also need to predict relevant articles and term of imprisonment.

   3) this is a multi-label classification problem. given a fact, it may exists one or more than one accusations and serveral relevant articles.

   4) this also a imbalanced classification problem, while some labels have many data, other labels only have few data. to get best performance, you are not only

      to balanced precision and recall for a single label, but also need to balanced importance among different labels based on requirement or your evaluation matrix.


if you have any suggestion or problem, a better idea on how to slove this problem,find a bug, or want to find a team member, you are welcomed to can contact with me:

brightmart@hotmail.com. you can also commit your code to this repository.


1.Desc
-------------------------------------------------------------------------
this repository contain models that learn from law cases to make a prediction:

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

        Theft: 10051

        Smuggling, trafficking, transporting, and making drugs: 8872

        Intentional injury: 6377

        Robbery: 5020

        Fraud: 3536

        Bribery: 3496

        Provocation: 3290

        Dangerous driving: 2758

        Organization, coercion, seduction, shelter, and introduction of prostitution: 2647

        Manufacture, trafficking, and Disseminating Obscene Articles: 2617

        Take drugs for others: 2597

        Traffic accident: 2562

        Embezzlement: 2391

        Illegal possession, possession of firearms, ammunition: 2349

        Intentional homicide: 2282

        Opening casinos: 2259

        Illegal possession of drugs: 2203

        Occupation encroachment: 2197

        Rape: 2192

        Falsification, alteration, sale and purchase of official documents, documents and seals of state organs: 2153

  5) preprocess value for imprisonment.

     range of imprisonment from 0 to 300(=12*25), it is raw value too big, not may lead to less efficient for model to learn.

     we will normalize imprisonment using following format:

          imprisonment= (imprisonment- imprisonment_mean)/imprisonment_std

     where imprisonment_mean stand for mean of imprisonment, imprisonment_std stand for stand deviation of imprisonment.

     we can easily to get its mean:, 26.2, and std:33.5 from training data. during test we will re-scale value back:

         imprisonment_test=(imprisonment_test+imprisonment_mean)*imprisonment_std

  6) normalize value of money:

     there are lots of int and float in this data related to money. such as 12343 or 3446.56, to make it easy for the model to learn it, we normalize

     int and float. for example to 10000 or 3000. check method replace_money_value in data_util.py

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

<img src="https://github.com/brightmart/ai_law/blob/master/data/f1_micro_macro.jpg"  width="60%" height="60%" />


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


4.Imbalance Classification for Skew Data
-------------------------------------------------------------------------

  since some labels associate with many data, others may only contain few data. we will try to use sampling to handle this problem.

  over sampling is used:

  we will first get frequency of each label in accusation and relevant article. given a input, we will get label of accusation and relevant article, and there

  frequency, then compute average value. we also set a threshold(e.g. 1500), add compute how many copy should we have in our training data(num_copy=threshold/average).

  e.g. below is what we did:

  input1:('freq_accusation:', 1230, 'freq_article:', 2973, ';num_copy:', 1)

  input2:('freq_accusation:', 167, 'freq_article:', 3525, ';num_copy:', 1)

  input3:('freq_accusation:', 282, 'freq_article:', 304, ';num_copy:', 5)

  input4:('freq_accusation:', 225, 'freq_article:', 22, ';num_copy:', 12)

  input5:('freq_accusation:', 489, 'freq_article:', 487, ';num_copy:', 3)

  input6:('freq_accusation:', 1134, 'freq_article:', 1148, ';num_copy:', 1)


  why we take average value for frequency of accusation and relevant article?

      as accsuation and relevant articles are related in many cases, we will use average

  check transform_data_to_index() under data_util.py



5.Transfer Learning & Pretrained Word Embedding
-------------------------------------------------------------------------
  download pretrained word embedding from https://github.com/Embedding/Chinese-Word-Vectors and enable flag 'use_embedding' during training.

  or download from https://pan.baidu.com/s/1o7MWrnc, password: wzqv , choose '.bin' file, embedding size is 64.

  command to import word embedding, especially for words contain chinese:

  import gensim

  from gensim.models import KeyedVectors

  word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True, unicode_errors='ignore')



6.Models
-------------------------------------------------------------------------
  1) HAN: hierarchical attention network(completed)

        embedding-->word level bi-lstm encoder-->word level attention-->sentence level bi-lstm encoder-->sentence level attention

  2) TextCNN(multi-layers)

       embedding-->CNN1(BN-->Relu)--->CNN2(BN-->Relu)--->Max-pooling

  3) DPCNN: deep pyramid cnn for text categorization

      text region embedding-->CNN1.1-->CNN2.1-->(Pooling/2-->CONV-->CONV)*N-->Pooling

  3) c-gru: CNN followed by GRU

       embedding--->CNN(BN-->Relu)--->bi-GRU

  4) gru-c: GRU followed by CNN

       embedding-->bi-GRU--->CNN(BN--->Relu)

  5) simple_pooling

    a) embedding-->max over each dimension of word embedding

    b) embedding-->mean over each dimension of word embedding

    c) embedding-->concat of max pooling and mean pooling

    d) embedding-->mean over each context(context is define as words around itself)-->max over context for each dimension.

  6) self-attention(transformer) TODO

  7) Convolutional Sequence to Sequence Learning TODO



7.Performance
-------------------------------------------------------------------------
   performance on validation dataset(seperate from training data) train for first 1000 steps(around one epoch):

   ('1.Accasation Score:', 70.8571253641409, ';2.Article Score:', 71.28389708554336, ';3.Penalty Score:', 69.78947368421059, ';Score ALL:', 211.93049613389485)

![alt text](https://github.com/brightmart/ai_law/blob/master/data/han_1000steps.jpg)


Performance on test env(small data, 155k training data),online:

Model |Accasation Score | Relevant Score | Penalty Score
--|--|--|--|
HAN|77.63 | 75.29 | 52.65
TextCNN(multiple layers)|79.91 | 76.87 | 53.62
c-gru| | |
gru-c| | |



Performance on test env(big data, 1.5 million training data),online:

Model |Accasation Score | Relevant Score | Penalty Score | Total Score
--|--|--|--|--|
TextCNN-multiple layers(online)|84.51 | 82.20 | 67.60 | 234.31
Deep Pyramid CNN(offline)|89.0 | 86.4 | 78.6 | 254
Hierarchical Attention Network(offline)|85.1 | 84.0 | 79.2 | 248.3

Notice: offline score is lower than online score for about 4.0.

 89.03954996862663, ';2.Article Score:', 86.38077500531911, ';3.Penalty Score:', 78.64466689362311, ';Score ALL:', 254.06499186756886)


8.Error Analysis
----------------------------------------------------------------
   TODO


9.Usage
-------------------------------------------------------------------------
  train:

     python HAN_train.py

    it will report macro f1 score and micro f1 score when doing validation, and save checkpoint to predictor/checkpoint

    optional parameters:

    --model: the name of model you will use. {han,text_cnn,dp_cnn,c_gru,c_gru2,gru,pooling} [han]

    --use_pretrained_embedding: whether use pretrained embedding or not. download it as discussed on section #5, otherwise set it to False. {True,False} [True]

    --embed_size: embedding size

    --hidden_size: hidden size

  predict:

     python3 main.py

  zip your model so that you can upload for testing purpose, run:

     zip -r ai_law_predictor.zip predictor  ===>it will zip all resources in directory predictor/ as a zip file.



10.Environment
-------------------------------------------------------------------------
   python 3 + tensorflow 1.8

   for people use python2, you can just add below lines:

      import sys

      reload(sys)

      sys.setdefaultencoding('utf-8')


11.Model Details
-------------------------------------------------------------------------
1).Hierarchical Attention Network:

    Implementation of Hierarchical Attention Networks for Document Classification

    Structure:

    embedding

    Word Encoder: word level bi-directional GRU to get rich representation of words

    Word Attention:word level attention to get important information in a sentence

    Sentence Encoder: sentence level bi-directional GRU to get rich representation of sentences

    Sentence Attention: sentence level attention to get important sentence among sentences

    One Layer MLP for transform document representation to each sub task's features.

    FC+Softmax


<img src="https://github.com/brightmart/ai_law/blob/master/data/model_law.jpg"  width="60%" height="60%" />


<img src="https://github.com/brightmart/text_classification/blob/master/images/HAN.JPG"  width="60%" height="60%" />


    check inference_han method from HAN_model.py under directory of predictor


2).TextCNN(Multiple Layers):

    Implementation of <a href="http://www.aclweb.org/anthology/D14-1181"> Convolutional Neural Networks for Sentence Classification </a>

    Structure: embedding-->CNN1(BN-->Relu)--->CNN2(BN-->Relu)--->Max-pooling-->concat features--->Fully Connected Layer

    In order to get very good result with TextCNN, you also need to read carefully about this paper <a href="https://arxiv.org/abs/1510.03820">A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification</a>: it give you some insights of things that can affect performance. although you need to  change some settings according to your specific task.

    Convolutional Neural Network is main building box for solve problems of computer vision. Now we will show how CNN can be used for NLP, in in particular, text classification. Sentence length will be different from one to another. So we will use pad to get fixed length, n. For each token in the sentence, we will use word embedding to get a fixed dimension vector, d. So our input is a 2-dimension matrix:(n,d). This is similar with image for CNN.

    Firstly, we will do convolutional operation to our input. It is a element-wise multiply between filter and part of input. We use k number of filters, each filter size is a 2-dimension matrix (f,d). Now the output will be k number of lists. Each list has a length of n-f+1. each element is a scalar. Notice that the second dimension will be always the dimension of word embedding. We are using different size of filters to get rich features from text inputs. And this is something similar with n-gram features.

    Secondly, we will do max pooling for the output of convolutional operation. For k number of lists, we will get k number of scalars.

    Thirdly, we will concatenate scalars to form final features. It is a fixed-size vector. And it is independent from the size of filters we use.

    Finally, we will use linear layer to project these features to per-defined labels.

<img src="https://github.com/brightmart/ai_law/blob/master/data/text_cnn_multiplelayers.jpg"  width="60%" height="60%" />


<img src="https://github.com/brightmart/text_classification/blob/master/images/TextCNN.JPG"  width="60%" height="60%" />



    check inference_text_cnn method from HAN_model.py under directory of predictor


3)DPCNN: Deep Pyramid CNN

   <a href='http://www.aclweb.org/anthology/P/P17/P17-1052.pdf'>Deep Pyramid Convolutional Neural Networks for Text Categorization</a>

    text region embedding-->CNN1.1-->CNN2.1-->(Pooling/2-->CONV-->CONV)*N-->Pooling

    this model is used for text categorization, you can think it is a text classification model with text length is quite long.

    basicly, it is a deep convolutional neural networks with repeat of building block: max-pooling and multiple layers of CNN.

    to make it easy for train this deep model, it also used skip connection as ResNet does. different from other deep models,

    conventionally when you reduce space size(input size for each layer), we will also increase depth of channels, so that computation resource in each layer is fixed.

    in deep pyramid CNN, it keep depth of channels, and gradually recuce space size(input size).

    main features of the model:

    a.downsampling with the number of feature maps fixed ===> reduce computation. as total computation time is twice

    the computation time of a single block.

    b. shortcut connections with pre-activation ====> so that it can train very deep neural network, similiar with ResNet.

    c. no need for dimension matching ===> although skip connection is used, but dimension is fixed(e.g. 250).

    d. text region embedding: embedding of a region of text convering one or more words.


<img src="https://github.com/brightmart/ai_law/blob/master/data/DPCNN.jpg"  width="50%" height="50%" />

<img src="https://github.com/brightmart/ai_law/blob/master/data/deep_pyramid_compare.jpg"  width="70%" height="70%" />




China law research cup judicial artificial intelligence challenge:
-------------------------------------------------------------------------

    Task 1 crime prediction: predict the accused's conviction according to the description and facts of the case in the criminal legal documents;

    Task 2 Recommendation of the relevant law: to predict the relevant laws in this case according to the description and facts of the case in the

    criminal legal documents;

    Task 3 time of imprisonment:  predict the defendant's sentence length.

    Data set including 2.68 million criminal law legal documents, criminal 183, relevant article 202 of law, sentence length including

     0-25 years, life imprisonment, sentence to death.



12.TODO
-------------------------------------------------------------------------
   0) normalize numbers for money to standard format or to some range

   1) tracking miss match of micro and macro of f1 score: balance micro and macro to maximize final f1 score

   2) error analysis: print and analysis error cases for each task, and get insight for improvement

   3) truncate or pad sequences in the beginning, or reverse(DONE)

   4) preprocess document as serveral sentences before graph model

   5) try pure CNN or attention models to speed up training(CNN DONE)


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

  4) <a href='https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf'>Hierarchical Attention Networks for Document Classification</a>

  5) Baseline Needs More Love: On Simple Word-Embedding-Based Modles and Associated Pooling Mechanisms

  6) <a href='http://www.aclweb.org/anthology/P/P17/P17-1052.pdf'>Deep Pyramid Convolutional Neural Networks for Text Categorization</a>

if you are smart or can contribute new ideas, join with us.

to be continued. for any problem, contact brightmart@hotmail.com

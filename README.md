# Short-Text-Classification

In order to facilitate research of Chinese short text classification (STC), we public two large-scale corpus of Chinese short text as the benchmark. Its details as follows.

# 1, BSM, 
a large-scale bullet screen messages of size 100K collected from short-video websites, where the message is used for binary sentiment classification - Positive or Negative. We randomly divided in 70% for training, 15% for validation and 15% for testing.

# 2, VSQ, 
a collection of search queries of size 150K gathered from publicly accessible video websites. It involves classifying a query into fine-grained 26 labels, such as film, travel, music, fashion and sports. The split for training, validation and testing are same as BSM.

# 3, Summary statistics of available 13 short text datasets.
Note that 1K is equal to 1000.
 Dataset | \#Class  | Avg.Len | Size(K) | \#Words | \#Train | \#Val | \#Test
 --------| :-----------:  | :-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:
 MR[1] | 2 | 21.0  |10.6| 21,384| 7,464| 1,599| 1,599
 SST-1|5|18.4   |11.8 |17,836| 8,544| 2,210| 1,101
 SST-2 | 2 |18.5 |9.6 |16,188| 6,920| 872| 1,821
 SUBJ [2]|2 |23.3| 10 |21,301 |7,001 |1,495|1，499
 TREC [3] |6|9.8 |6.4| 8，764 |5，452 |500 |500
 CR [4]| 2 |18.8| 3.7 |5，339| 2，639| 566 |565
 MPQA [5] |2|3.1|10.6|6，246|7，423|1，590|1，590
 Snippets [6]|8|14.5|12.3|29,040|8,368|1,851|1,851
 Ohsumed [7]|23|6.8|7.4|11,764|5,180|1,110|1,110
 Twitter|2|3.5|10|21,065|7,000|1,500|1,500
 TagMyNews [8]|7|5.1|32.5|38,629|22,785|4,882|4,882
 <b>DanMU(or BSM)</b>[9,10] | 2|6.3|100|4,560|<b>70K</b>|<b>15K</b>|<b>15K</b>
 <b>CVQD(or VSQ)</b>[9,10] | 26 |8.5|150|5,320|<b>105K</b>|<b>22.5K</b>|<b>22.5K</b>
 
# Please cite thehe References if you find it helpful. Thanks
[9] @InProceedings{PathWalk,<br>
  author =  {Wei Pang},<br>
  title = {Within-Dataset Graph Enhancement for Short Text Classification},<br>
  booktitle = {MLG of ECML-PKDD},<br>
  year = {2022}<br>
}<br>

[10] @Article{TermGraph,<br>
  author =  {Wei Pang},<br>
  title = {Short Text Classification via Term Graph},<br>
  journal = {arXiv:2001.10338},<br>
  year = {2020}<br>
}<br>

[1] B. Pang, L. Lee, Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales, in: ACL, 2005, pp. 115–124<br>
[2] B. Pang, L. Lee, A sentimental education: Sentiment analysis using sub- jectivity summarization based on minimum cuts, in: ACL, 2004.<br>
[3] X. Li, D. Roth, Learning question classifiers, in: ACL, 2002 <br>
[4] M. Hu, B. Liu, Mining and summarizing customer reviews, in: KDD, 2004<br>
[5] L. Deng, J. Wiebe, Mpqa 3.0: Entity/event-level sentiment corpus, in: NAACL, 2015, pp. 1323–1328<br>
[6] X.-H. Phan, L.-M. Nguyen, S. Horiguchi, Learning to classify short and sparse text and web with hidden topics from large-scale data collections, in: WWW, 2008, pp. 91–100 <br>
[7] W. Hersh, C. Buckley, T. Leone, D. Hickam, Ohsumed: An interactive retrieval evaluation and new large test collection for research, in: SIGIR, 1994, pp. 192–201<br>
[8] D. Vitale, P. Ferragina, U. Scaiella, Classification of short texts by deploy- ing topical annotations, in: ECIR, 2012, pp. 376–387<br>




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
 MR | 2 | 21.0  |10.6| 21,384| 7,464| 1,599| 1,599
 SST-1|5|18.4   |11.8 |17,836| 8,544| 2,210| 1,101
 SST-2 | 2 |18.5 |9.6 |16,188| 6,920| 872| 1,821
 SUBJ|2 |23.3| 10 |21,301 |7,001 |1,495|1，499
 TREC|6|9.8 |6.4| 8，764 |5，452 |500 |500
 CR| 2 |18.8| 3.7 |5，339| 2，639| 566 |565
 MPQA|2|3.1|10.6|6，246|7，423|1，590|1，590
 Snippets|8|14.5|12.3|29,040|8,368|1,851|1,851
 Ohsumed|23|6.8|7.4|11,764|5,180|1,110|1,110
 Twitter|2|3.5|10|21,065|7,000|1,500|1,500
 TagMyNews|7|5.1|32.5|38,629|22,785|4,882|4,882
 <b>BSM</b> | 2|6.3|100|4,560|<b>70K</b>|<b>15K</b>|<b>15K</b>
 <b>VSQ</b> | 26 |8.5|150|5,320|<b>105K</b>|<b>22.5K</b>|<b>22.5K</b>
 


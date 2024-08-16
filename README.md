# 中文短文本分类（Chinese Short-Text-Classification）
为支持中文短文本分类（STC）研究，我们发布了两个大规模的中文短文本语料库作为基准测试。详细信息如下。如果您认为此项目有价值，请考虑给我们点个星⭐。<br>
To support research in Chinese short text classification (STC), we have released two large-scale Chinese short text corpora as benchmarks. Details are provided below. If you find this project valuable, please consider leaving a star⭐.<br>

# 开源中文短文本数据集 - Open-source Chinese short-text dataset
## 1, BSM
二分类情感识别，10万条中文弹幕消息，有两个情感类别：积极的和消极的，分成训练集（70%），验证集（15%）和测试集（15%）。<br>
A large-scale dataset of 100K bullet screen messages was collected from short-video platforms for binary sentiment classification (Positive/Negative). The dataset was randomly split into 70% for training, 15% for validation, and 15% for testing.

## 2, VSQ
中文视频搜索引擎的用户Query，包括26个Query分类类别，数据集划分同BSM。<br>
A dataset of 150K search queries was collected from publicly accessible video websites, classified into 26 fine-grained categories such as film, travel, music, fashion, and sports. The data is split into 70% for training, 15% for validation, and 15% for testing, consistent with the BSM dataset.

## 3, 13种短文本数据集统计信息，Summary statistics of available 13 short text datasets.
Note that 1K is equal to 1000.
 Dataset | \#Class  | Avg.Len | Size(K) | \#Words | \#Train | \#Val | \#Test
 --------| :-----------:  | :-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:
 MR[1] | 2 | 21.0  |10.6| 21,384| 7,464| 1,599| 1,599
 SST-1|5|18.4   |11.8 |17,836| 8,544| 2,210| 1,101
 SST-2 | 2 |18.5 |9.6 |16,188| 6,920| 872| 1,821
 SUBJ [2]|2 |23.3| 10 |21,301 |7,001 |1,495|1,499
 TREC [3] |6|9.8 |6.4| 8,764 |5,452 |500 |500
 CR [4]| 2 |18.8| 3.7 |5,339| 2,639| 566 |565
 MPQA [5] |2|3.1|10.6|6,246|7,423|1,590|1,590
 Snippets [6]|8|14.5|12.3|29,040|8,368|1,851|1,851
 Ohsumed [7]|23|6.8|7.4|11,764|5,180|1,110|1,110
 Twitter|2|3.5|10|21,065|7,000|1,500|1,500
 TagMyNews [8]|7|5.1|32.5|38,629|22,785|4,882|4,882
 <b>DanMU(or BSM)</b>[9,10] | 2|6.3|100|4,560|<b>70K</b>|<b>15K</b>|<b>15K</b>
 <b>CVQD(or VSQ)</b>[9,10] | 26 |8.5|150|5,320|<b>105K</b>|<b>22.5K</b>|<b>22.5K</b>

# PathWalk短文本分类训练代码 Training Codes
```
cd pathwalk
nohup bash run_training.sh > r.log 2>&1 &
```
# Citation
```
@Inproceedings{TermGraph,
    author = {Wei Pang and Duan Ruixue and Ning Li},
    title = {Within-Dataset Graph Enhancement for Short Text Classification},
    booktitle = {ECMLPKDD Workshop},
    year = {2022},
    pages = {},
    url = {https://openreview.net/forum?id=KOZTylWrOYu&referrer=%5Bthe%20profile%20of%20Wei%20Pang%5D(%2Fprofile%3Fid%3D~Wei_Pang4)},
}

@Article{ShortGraph,
  author = {Wei Pang},
  title = {Short Text Classification via Term Graph},
  journal = {arXiv.2001.10338},
  month = {Jan},
  year = {2020},
  pages = {},
  doi = {https://doi.org/10.48550/arXiv.2001.10338}
}
```

# References
[1] B. Pang, L. Lee, Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales, in: ACL, 2005, pp. 115–124<br>
[2] B. Pang, L. Lee, A sentimental education: Sentiment analysis using sub- jectivity summarization based on minimum cuts, in: ACL, 2004.<br>
[3] X. Li, D. Roth, Learning question classifiers, in: ACL, 2002 <br>
[4] M. Hu, B. Liu, Mining and summarizing customer reviews, in: KDD, 2004<br>
[5] L. Deng, J. Wiebe, Mpqa 3.0: Entity/event-level sentiment corpus, in: NAACL, 2015, pp. 1323–1328<br>
[6] X.-H. Phan, L.-M. Nguyen, S. Horiguchi, Learning to classify short and sparse text and web with hidden topics from large-scale data collections, in: WWW, 2008, pp. 91–100 <br>
[7] W. Hersh, C. Buckley, T. Leone, D. Hickam, Ohsumed: An interactive retrieval evaluation and new large test collection for research, in: SIGIR, 1994, pp. 192–201<br>
[8] D. Vitale, P. Ferragina, U. Scaiella, Classification of short texts by deploying topical annotations, in: ECIR, 2012, pp. 376–387<br>
[9] W. Pang et al., Within-Dataset Graph Enhancement for Short Text Classification, in ECML-PKDD 2022<br>
[10] Wei Pang. Short Text Classification via Term Graph, in arXiv:2001.10338, 2020<br>

# License
All datasets are licensed under Apache 2.0.



# Short-Text-Classification

In order to facilitate research of Chinese short text classification (STC), we public two large-scale corpus of Chinese short text as the benchmark. Its details as follows.

# 1, BSM, 
a large-scale bullet screen messages of size 100K collected from short-video websites, where the message is used for binary sentiment classification - Positive or Negative. We randomly divided in 70% for training, 15% for validation and 15% for testing.

# 2, VSQ, 
a collection of search queries of size 150K gathered from publicly accessible video websites. It involves classifying a query into fine-grained 26 labels, such as film, travel, music, fashion and sports. The split for training, validation and testing are same as BSM.

# 3, Summary statistics of available 13 short text datasets.
Note that 1K is equal to 1000.
 Dataset | \#Class  | Ang.Len | Size(K) | \#Words | \#Train | \#Val | \#Test
 --------| :-----------:  | :-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:
 guesser[20] | qgen[20] | 5  |41.6| 43.5| 47.1| 39.2| 40.8
 guesser(MN)[27]|TPG[27]|8   |- |48.77| -| -| -
 guesser[19] | qgen[19] |8 |- |44.6| -| -| -
 GST(ours)|VDST[13] (ours)|5 |<b>77.38</b>| <b>77.30</b> |<b>77.23</b> |<b>75.11</b> |<b>75.20</b>
 GST(ours)|VDST[13] (ours)|8 |<b>83.22</b>| <b>83.32</b> |<b>83.46</b> |<b>81.50</b> |<b>81.55</b>
 Human[19]| - |-| - |84.4| -| - |84.4


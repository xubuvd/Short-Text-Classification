


(vd) pangwei@gpu3:~/pathwalk/Chinese-Text-Classification$ nohup bash start.sh > hahahha_20220406.log 2>&1 &
[3] 3591834
(vd) pangwei@gpu3:~/pathwalk/Chinese-Text-Classification$ grep -n  "Test Acc:" log_*.log 
log_DPCNN_2022052803_CR.log:67:Test Loss:  0.52,  Test Acc: 75.75%
log_DPCNN_2022052803_CVQD.log:76:Test Loss:   1.5,  Test Acc: 56.94%
log_DPCNN_2022052803_DanMu.log:64:Test Loss:  0.17,  Test Acc: 93.75%
log_DPCNN_2022052803_MPQA.log:64:Test Loss:  0.48,  Test Acc: 79.43%
log_DPCNN_2022052803_MR.log:67:Test Loss:  0.62,  Test Acc: 66.23%
log_DPCNN_2022052803_SST1.log:73:Test Loss:   1.5,  Test Acc: 36.11%
log_DPCNN_2022052803_SST2.log:67:Test Loss:  0.59,  Test Acc: 68.81%
log_DPCNN_2022052803_SUBJ.log:70:Test Loss:  0.33,  Test Acc: 87.53%
log_DPCNN_2022052803_TREC.log:73:Test Loss:  0.38,  Test Acc: 87.20%
log_FastText_2022052803_CR.log:66:Test Loss:   0.5,  Test Acc: 75.93%
log_FastText_2022052803_CVQD.log:69:Test Loss:   1.5,  Test Acc: 58.21%
log_FastText_2022052803_DanMu.log:63:Test Loss:  0.15,  Test Acc: 94.51%
log_FastText_2022052803_MPQA.log:66:Test Loss:  0.44,  Test Acc: 81.70%
log_FastText_2022052803_MR.log:63:Test Loss:  0.68,  Test Acc: 60.60%
log_FastText_2022052803_SST1.log:69:Test Loss:   1.5,  Test Acc: 36.88%
log_FastText_2022052803_SST2.log:63:Test Loss:  0.57,  Test Acc: 70.68%
log_FastText_2022052803_SUBJ.log:63:Test Loss:  0.25,  Test Acc: 88.99%
log_FastText_2022052803_TREC.log:72:Test Loss:  0.31,  Test Acc: 88.20%
log_TextCNN_2022052803_CR.log:66:Test Loss:  0.56,  Test Acc: 75.75%
log_TextCNN_2022052803_CVQD.log:66:Test Loss:   1.5,  Test Acc: 57.34%
log_TextCNN_2022052803_DanMu.log:63:Test Loss:  0.18,  Test Acc: 93.37%
log_TextCNN_2022052803_MPQA.log:63:Test Loss:  0.48,  Test Acc: 79.87%
log_TextCNN_2022052803_MR.log:63:Test Loss:  0.67,  Test Acc: 62.98%
log_TextCNN_2022052803_SST1.log:66:Test Loss:   1.6,  Test Acc: 31.40%
log_TextCNN_2022052803_SST2.log:63:Test Loss:  0.63,  Test Acc: 68.15%
log_TextCNN_2022052803_SUBJ.log:63:Test Loss:  0.36,  Test Acc: 83.92%
log_TextCNN_2022052803_TREC.log:69:Test Loss:  0.38,  Test Acc: 89.40%
log_TextLSTM_2022052803_CR.log:65:Test Loss:  0.55,  Test Acc: 72.74%
log_TextLSTM_2022052803_CVQD.log:77:Test Loss:   1.6,  Test Acc: 55.20%
log_TextLSTM_2022052803_DanMu.log:62:Test Loss:  0.15,  Test Acc: 94.45%
log_TextLSTM_2022052803_MPQA.log:68:Test Loss:  0.46,  Test Acc: 80.50%
log_TextLSTM_2022052803_MR.log:62:Test Loss:  0.58,  Test Acc: 70.92%
log_TextLSTM_2022052803_SST1.log:68:Test Loss:   1.4,  Test Acc: 35.38%
log_TextLSTM_2022052803_SST2.log:62:Test Loss:   0.5,  Test Acc: 75.56%
log_TextLSTM_2022052803_SUBJ.log:65:Test Loss:   0.3,  Test Acc: 89.26%
log_TextLSTM_2022052803_TREC.log:80:Test Loss:  0.41,  Test Acc: 88.00%
log_TextRCNN_2022052803_CR.log:62:Test Loss:  0.46,  Test Acc: 78.23%
log_TextRCNN_2022052803_CVQD.log:71:Test Loss:   1.6,  Test Acc: 55.48%
log_TextRCNN_2022052803_DanMu.log:62:Test Loss:  0.18,  Test Acc: 94.17%
log_TextRCNN_2022052803_MPQA.log:62:Test Loss:  0.46,  Test Acc: 80.25%
log_TextRCNN_2022052803_MR.log:62:Test Loss:  0.65,  Test Acc: 66.98%
log_TextRCNN_2022052803_SST1.log:65:Test Loss:   1.5,  Test Acc: 32.94%
log_TextRCNN_2022052803_SST2.log:62:Test Loss:  0.55,  Test Acc: 73.59%
log_TextRCNN_2022052803_SUBJ.log:62:Test Loss:  0.31,  Test Acc: 87.66%
log_TextRCNN_2022052803_TREC.log:65:Test Loss:  0.37,  Test Acc: 89.20%
log_TextRNN_2022052803_CR.log:62:Test Loss:  0.54,  Test Acc: 72.92%
log_TextRNN_2022052803_CVQD.log:68:Test Loss:   1.6,  Test Acc: 53.87%
log_TextRNN_2022052803_DanMu.log:62:Test Loss:  0.16,  Test Acc: 94.20%
log_TextRNN_2022052803_MPQA.log:71:Test Loss:  0.46,  Test Acc: 81.19%
log_TextRNN_2022052803_MR.log:59:Test Loss:   0.7,  Test Acc: 50.34%
log_TextRNN_2022052803_SST1.log:68:Test Loss:   1.5,  Test Acc: 36.29%
log_TextRNN_2022052803_SST2.log:65:Test Loss:  0.52,  Test Acc: 76.11%
log_TextRNN_2022052803_SUBJ.log:62:Test Loss:  0.29,  Test Acc: 87.66%
log_TextRNN_2022052803_TREC.log:68:Test Loss:  0.41,  Test Acc: 86.60%
log_TextRNN_Att_2022052803_CR.log:69:Test Loss:  0.47,  Test Acc: 78.23%
log_TextRNN_Att_2022052803_CVQD.log:93:Test Loss:   1.5,  Test Acc: 56.67%
log_TextRNN_Att_2022052803_DanMu.log:75:Test Loss:  0.15,  Test Acc: 94.62%
log_TextRNN_Att_2022052803_MPQA.log:63:Test Loss:  0.44,  Test Acc: 80.94%
log_TextRNN_Att_2022052803_MR.log:63:Test Loss:   0.6,  Test Acc: 67.98%
log_TextRNN_Att_2022052803_SST1.log:69:Test Loss:   1.4,  Test Acc: 38.91%
log_TextRNN_Att_2022052803_SST2.log:66:Test Loss:  0.47,  Test Acc: 78.03%
log_TextRNN_Att_2022052803_SUBJ.log:63:Test Loss:  0.26,  Test Acc: 88.66%
log_TextRNN_Att_2022052803_TREC.log:66:Test Loss:   0.3,  Test Acc: 89.20%
log_Transformer_2022052803_CR.log:114:Test Loss:  0.67,  Test Acc: 63.36%
log_Transformer_2022052803_DanMu.log:138:Test Loss:  0.19,  Test Acc: 93.40%
log_Transformer_2022052803_MPQA.log:132:Test Loss:  0.49,  Test Acc: 79.37%
log_Transformer_2022052803_MR.log:120:Test Loss:  0.68,  Test Acc: 57.66%
log_Transformer_2022052803_SST1.log:123:Test Loss:   1.6,  Test Acc: 30.05%
log_Transformer_2022052803_SST2.log:120:Test Loss:  0.62,  Test Acc: 65.46%
log_Transformer_2022052803_SUBJ.log:117:Test Loss:  0.38,  Test Acc: 83.72%
log_Transformer_2022052803_TREC.log:123:Test Loss:  0.39,  Test Acc: 86.40%

(vd) pangwei@gpu3:~/pathwalk/Chinese-Text-Classification$ nohup bash start.sh > s.log 2>&1 &
[1] 1443615



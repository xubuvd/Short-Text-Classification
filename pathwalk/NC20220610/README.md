
(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_pathwalk_2022022701_34gpu1_15ep64bs.log 2>&1 &
[1] 1168108
a, view a short-text as a graph path

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_THUCNews_2022022702_34gpu1_15ep32bs.log 2>&1 &
[1] 4003, view a short-text as a graph path
embedding_size: 128
lstm_hidden_size: 128
epoch:1	Acc:0.7104	loss_epoch:1.753352403640747
epoch:2	Acc:0.7735	loss_epoch:1.687174916267395
epoch:3	Acc:0.784	loss_epoch:1.6753358840942383
epoch:4	Acc:0.8137	loss_epoch:1.6468651294708252

embedding_size: 300
lstm_hidden_size: 300
parameters numbers:169766410
epoch:1	Acc:0.7484	loss_epoch:1.7129709720611572	lr:0.0009998578030572343
epoch:2	Acc:0.7925	loss_epoch:1.6690400838851929	lr:0.001
epoch:3	Acc:0.8116	loss_epoch:1.648537516593933	lr:0.001
epoch:4	Acc:0.8324	loss_epoch:1.628994107246399	lr:0.0004
epoch:5	Acc:0.8366	loss_epoch:1.6228269338607788	lr:0.0004
epoch:6	Acc:0.8396	loss_epoch:1.6211469173431396	lr:0.0004
epoch:7	Acc:0.8426	loss_epoch:1.6177127361297607	lr:0.0004
epoch:8	Acc:0.8479	loss_epoch:1.613599419593811	lr:0.00016000000000000004
epoch:9	Acc:0.847	loss_epoch:1.6134850978851318	lr:0.00016000000000000004
epoch:10	Acc:0.8486	loss_epoch:1.612518072128296	lr:0.00016000000000000004
epoch:11	Acc:0.8485	loss_epoch:1.6122890710830688	lr:0.00016000000000000004
epoch:12	Acc:0.8488	loss_epoch:1.6124022006988525	lr:0.00016000000000000004
epoch:13	Acc:0.8482	loss_epoch:1.6125837564468384	lr:0.00016000000000000004
epoch:14	Acc:0.8487	loss_epoch:1.6115041971206665	lr:6.400000000000001e-05
epoch:15	Acc:0.8494	loss_epoch:1.612001657485962	lr:6.400000000000001e-05

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_THUCNews_2022022703_34gpu0_15ep32bs.log 2>&1 &
[1] 5886
embedding_size: 512
lstm_hidden_size: 512
parameters numbers:294293514
epoch:1	Acc:0.7506	loss_epoch:1.7076743841171265	lr:0.0009998578030572343
epoch:2	Acc:0.7843	loss_epoch:1.6749584674835205	lr:0.001
epoch:3	Acc:0.8098	loss_epoch:1.6504188776016235	lr:0.001
epoch:4	Acc:0.8303	loss_epoch:1.6292186975479126	lr:0.0004
epoch:5	Acc:0.8362	loss_epoch:1.6254733800888062	lr:0.0004
epoch:6	Acc:0.8419	loss_epoch:1.6191141605377197	lr:0.0004
epoch:7	Acc:0.8427	loss_epoch:1.6168303489685059	lr:0.0004
epoch:8	Acc:0.8485	loss_epoch:1.612267255783081	lr:0.00016000000000000004
epoch:9	Acc:0.8483	loss_epoch:1.6123583316802979	lr:0.00016000000000000004
epoch:10	Acc:0.8497	loss_epoch:1.6109888553619385	lr:0.00016000000000000004
epoch:11	Acc:0.8513	loss_epoch:1.6095577478408813	lr:0.00016000000000000004
epoch:12	Acc:0.848	loss_epoch:1.6118266582489014	lr:0.00016000000000000004

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_THUCNews_2022022704_34gpu1_15ep32bs.log 2>&1 &
[1] 2596
a, adds dropout=0.4 in BiLSTM encoder;
b, sets dropout_fc=0.2 in linear layers;
embedding_size: 384
lstm_hidden_size: 384
parameters numbers:218655754
epoch:1	Acc:0.7425	loss_epoch:1.716838002204895	lr:0.0009998578030572343
epoch:2	Acc:0.7866	loss_epoch:1.6726537942886353	lr:0.001
epoch:3	Acc:0.8059	loss_epoch:1.6532814502716064	lr:0.001
epoch:4	Acc:0.8312	loss_epoch:1.6296308040618896	lr:0.0004
epoch:5	Acc:0.834	loss_epoch:1.625578761100769	lr:0.0004
epoch:6	Acc:0.8367	loss_epoch:1.6229355335235596	lr:0.0004
epoch:7	Acc:0.8438	loss_epoch:1.6165691614151	lr:0.0004
epoch:8	Acc:0.8493	loss_epoch:1.6116479635238647	lr:0.00016000000000000004
epoch:9	Acc:0.8499	loss_epoch:1.6115247011184692	lr:0.00016000000000000004
epoch:10	Acc:0.8487	loss_epoch:1.6113677024841309	lr:0.00016000000000000004
epoch:11	Acc:0.851	loss_epoch:1.6099953651428223	lr:0.00016000000000000004
epoch:12	Acc:0.8483	loss_epoch:1.610576868057251	lr:0.00016000000000000004
epoch:13	Acc:0.8495	loss_epoch:1.6104605197906494	lr:0.00016000000000000004
epoch:14	Acc:0.8515	loss_epoch:1.608949065208435	lr:6.400000000000001e-05
epoch:15	Acc:0.8523	loss_epoch:1.6084716320037842	lr:6.400000000000001e-05

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_THUCNews_2022022705_34gpu1_15ep32bs.log 2>&1 &
[1] 3345862,
a, it is a purely BiLSTM version, called TextRNN model, without using graph path;
epoch:1	Acc:0.7545	loss_epoch:1.7065314054489136	lr:0.0009998578030572343
epoch:2	Acc:0.7888	loss_epoch:1.6720250844955444	lr:0.001
epoch:3	Acc:0.8002	loss_epoch:1.6596468687057495	lr:0.001
epoch:4	Acc:0.8253	loss_epoch:1.6350855827331543	lr:0.0004
epoch:5	Acc:0.8293	loss_epoch:1.630234718322754	lr:0.0004
epoch:6	Acc:0.8366	loss_epoch:1.6246373653411865	lr:0.0004
epoch:7	Acc:0.8385	loss_epoch:1.6218327283859253	lr:0.0004
epoch:8	Acc:0.8436	loss_epoch:1.6168047189712524	lr:0.00016000000000000004
epoch:9	Acc:0.8431	loss_epoch:1.6168767213821411	lr:0.00016000000000000004
epoch:10	Acc:0.8439	loss_epoch:1.6165332794189453	lr:0.00016000000000000004
epoch:11	Acc:0.8449	loss_epoch:1.6155866384506226	lr:0.00016000000000000004
epoch:12	Acc:0.8461	loss_epoch:1.6145989894866943	lr:0.00016000000000000004
epoch:13	Acc:0.8482	loss_epoch:1.6127018928527832	lr:0.00016000000000000004
epoch:14	Acc:0.8484	loss_epoch:1.6125253438949585	lr:6.400000000000001e-05
epoch:15	Acc:0.8496	loss_epoch:1.6112762689590454	lr:6.400000000000001e-05

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_THUCNews_2022022706_34gpu0_15ep32bs.log 2>&1 &
[2] 1729539, 50 epoch, graph path
epoch:1	Acc:0.7469	loss_epoch:1.7128876447677612	lr:0.0009998578030572343
epoch:2	Acc:0.7887	loss_epoch:1.6716171503067017	lr:0.001
epoch:3	Acc:0.8083	loss_epoch:1.651634693145752	lr:0.001
epoch:4	Acc:0.8321	loss_epoch:1.6286931037902832	lr:0.0004
epoch:5	Acc:0.8402	loss_epoch:1.6196376085281372	lr:0.0004
epoch:6	Acc:0.8415	loss_epoch:1.6184265613555908	lr:0.0004
epoch:7	Acc:0.8453	loss_epoch:1.6158264875411987	lr:0.0004
epoch:8	Acc:0.849	loss_epoch:1.6107679605484009	lr:0.00016000000000000004
epoch:9	Acc:0.8517	loss_epoch:1.608981966972351	lr:0.00016000000000000004
epoch:10	Acc:0.8532	loss_epoch:1.6075520515441895	lr:0.00016000000000000004
epoch:11	Acc:0.8514	loss_epoch:1.6087851524353027	lr:0.00016000000000000004
epoch:12	Acc:0.8528	loss_epoch:1.6078330278396606	lr:0.00016000000000000004
epoch:49	Acc:0.855	loss_epoch:1.6052296161651611	lr:6.400000000000001e-05
epoch:50	Acc:0.8549	loss_epoch:1.6050770282745361	lr:6.400000000000001e-05

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_THUCNews_2022022707_34gpu0_15ep32bs.log 2>&1 &
[1] 68245, 15 epoch, 128 batch_size; purely TextRNN model, with only one layer in self.encode_linear;
epoch:1	Acc:0.7264	loss_epoch:1.735791563987732	lr:0.001
epoch:2	Acc:0.7721	loss_epoch:1.69033682346344	lr:0.001
epoch:3	Acc:0.7877	loss_epoch:1.6730762720108032	lr:0.0004
epoch:4	Acc:0.8092	loss_epoch:1.6545922756195068	lr:0.0004
epoch:5	Acc:0.814	loss_epoch:1.6466425657272339	lr:0.0004
epoch:6	Acc:0.819	loss_epoch:1.6430542469024658	lr:0.0004
epoch:7	Acc:0.8201	loss_epoch:1.640801191329956	lr:0.00016000000000000004
epoch:8	Acc:0.8258	loss_epoch:1.6334065198898315	lr:0.00016000000000000004
epoch:9	Acc:0.8292	loss_epoch:1.6332056522369385	lr:0.00016000000000000004
epoch:10	Acc:0.828	loss_epoch:1.6321407556533813	lr:0.00016000000000000004
epoch:11	Acc:0.8286	loss_epoch:1.6299306154251099	lr:0.00016000000000000004
epoch:12	Acc:0.8322	loss_epoch:1.6307005882263184	lr:0.00016000000000000004
epoch:13	Acc:0.8323	loss_epoch:1.6299254894256592	lr:6.400000000000001e-05
epoch:14	Acc:0.8319	loss_epoch:1.6279716491699219	lr:6.400000000000001e-05
epoch:15	Acc:0.8321	loss_epoch:1.6288868188858032	lr:6.400000000000001e-05


(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_THUCNews_202202270_34gpu0_15ep32bs.log 2>&1 &
[1] 71975, 15 epoch, 128 batch_size; purely TextRNN model, with 5 layers in self.encode_linear;
epoch:1	Acc:0.7559	loss_epoch:1.7044199705123901	lr:0.001
epoch:2	Acc:0.7946	loss_epoch:1.6668274402618408	lr:0.001
epoch:3	Acc:0.8029	loss_epoch:1.6597681045532227	lr:0.0004
epoch:4	Acc:0.8244	loss_epoch:1.6362944841384888	lr:0.0004
epoch:5	Acc:0.8288	loss_epoch:1.6297659873962402	lr:0.0004
epoch:6	Acc:0.8313	loss_epoch:1.6288031339645386	lr:0.0004
epoch:7	Acc:0.8378	loss_epoch:1.6213816404342651	lr:0.00016000000000000004
epoch:8	Acc:0.8414	loss_epoch:1.6194254159927368	lr:0.00016000000000000004
epoch:9	Acc:0.843	loss_epoch:1.6178492307662964	lr:0.00016000000000000004
epoch:10	Acc:0.8448	loss_epoch:1.6168237924575806	lr:0.00016000000000000004
epoch:11	Acc:0.8463	loss_epoch:1.6160105466842651	lr:0.00016000000000000004
epoch:12	Acc:0.8457	loss_epoch:1.6164461374282837	lr:0.00016000000000000004
epoch:13	Acc:0.8463	loss_epoch:1.61501944065094	lr:6.400000000000001e-05
epoch:14	Acc:0.847	loss_epoch:1.6130884885787964	lr:6.400000000000001e-05
epoch:15	Acc:0.8472	loss_epoch:1.6145505905151367	lr:6.400000000000001e-05
It shows that self.encode_linear with 5 layers is better than one layer;

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_THUCNews_2022022709_34gpu1_15ep32bs.log 2>&1 &
[1] 7495, graph path model with 5 layers in self.encode_linear;

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_THUCNews_2022022711_34gpu1_15ep32bs_placeholde.log 2>&1 &
[1] 2755, placeholder

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_THUCNews_2022022712_34gpu0_15ep32bs_placeholde.log 2>&1 &
[2] 4176, placeholder


(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_THUCNews_2022030301_34gpu1_15ep128bs.log 2>&1 &
[1] 2573,
128 embedding size,
128 BiLSTM hidden state size,
15 epochs;
epoch:1	Acc:0.5626	loss_epoch:1.9103807210922241	lr:0.0003
epoch:2	Acc:0.6777	loss_epoch:1.7853461503982544	lr:0.0003
epoch:3	Acc:0.7238	loss_epoch:1.7398813962936401	lr:0.00011999999999999999
epoch:4	Acc:0.7417	loss_epoch:1.7198246717453003	lr:0.00011999999999999999
epoch:5	Acc:0.7534	loss_epoch:1.7083892822265625	lr:0.00011999999999999999
epoch:6	Acc:0.7599	loss_epoch:1.7007222175598145	lr:0.00011999999999999999
epoch:7	Acc:0.7679	loss_epoch:1.6927026510238647	lr:4.800000000000001e-05
epoch:8	Acc:0.7711	loss_epoch:1.6900370121002197	lr:4.800000000000001e-05
epoch:9	Acc:0.7738	loss_epoch:1.6888340711593628	lr:4.800000000000001e-05
epoch:10	Acc:0.7746	loss_epoch:1.6852067708969116	lr:4.800000000000001e-05
epoch:11	Acc:0.7792	loss_epoch:1.68329918384552	lr:4.800000000000001e-05
epoch:12	Acc:0.7796	loss_epoch:1.6807570457458496	lr:4.800000000000001e-05
epoch:13	Acc:0.7812	loss_epoch:1.6799595355987549	lr:1.9200000000000003e-05
epoch:14	Acc:0.7809	loss_epoch:1.6775496006011963	lr:1.9200000000000003e-05
epoch:15	Acc:0.7822	loss_epoch:1.6793725490570068	lr:1.9200000000000003e-05


(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_DanMu_2022031401_34gpu1_20ep64bs.log 2>&1 &
[1] 904006,

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_CVQD_2022031402_34gpu0_20ep64bs.log 2>&1 &
[2] 907239

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_MR_2022031403_34gpu0_20ep64bs.log 2>&1 &
[1] 919394,
short text as a graph path;
epoch:1	Acc:0.5628517823639775	loss_epoch:0.6847895383834839	lr:0.001
epoch:17	Acc:0.5872420262664165	loss_epoch:0.7174856662750244	lr:2.5600000000000006e-05
epoch:20	Acc:0.5847404627892433	loss_epoch:0.7177119851112366	lr:2.5600000000000006e-05

short text as a bag of words commonly used:
epoch:1	Acc:0.5603502188868043	loss_epoch:0.6809850335121155	lr:0.001
epoch:9	Acc:0.6235146966854284	loss_epoch:0.6759173274040222	lr:0.00016000000000000004
epoch:18	Acc:0.6216385240775485	loss_epoch:0.678311288356781	lr:2.5600000000000006e-05
epoch:19	Acc:0.6241400875547217	loss_epoch:0.6781412363052368	lr:2.5600000000000006e-05
epoch:20	Acc:0.6216385240775485	loss_epoch:0.6791479587554932	lr:2.5600000000000006e-05
###introducing in-links between consecutive words to hurt the accuracy on dev set; a bad thing.

#### node represenation + in_link representation, BiLSTM( [n_{i} + e_{i,i+1}],....,)
epoch:1	Acc:0.5403377110694184	loss_epoch:0.6885731220245361	lr:0.001
epoch:16	Acc:0.6035021888680425	loss_epoch:0.6984019875526428	lr:6.400000000000001e-05
epoch:20	Acc:0.5959974984365228	loss_epoch:0.6997578144073486	lr:2.5600000000000006e-05
# it shows in_link representation improves the acc on dev set.

[1] 545453, node represenation + in_link representation + neighbourhood nodes and edges;
epoch:1	Acc:0.5453408380237649	loss_epoch:0.6919599175453186	lr:0.001
epoch:4	Acc:0.6028767979987493	loss_epoch:0.6909120678901672	lr:0.0004
epoch:14	Acc:0.6128830519074422	loss_epoch:0.6917475461959839	lr:6.400000000000001e-05
epoch:20	Acc:0.6097560975609756	loss_epoch:0.6917480230331421	lr:2.5600000000000006e-05
Test Loss:  0.69,  Test Acc: 61.06%

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_DanMu_2022031701_34gpu1_20ep64bs.log 2>&1 &
[1] 967847,
epoch:1	    Acc:0.8939636752136753	loss_epoch:0.4155022203922272	lr:0.0009992687385740402
epoch:15	Acc:0.9362980769230769	loss_epoch:0.3761737644672394	lr:6.400000000000001e-05
epoch:20	Acc:0.9358974358974359	loss_epoch:0.37626442313194275	lr:2.5600000000000006e-05
Test Loss:  0.38,  Test Acc: 93.15% # select_indegree_num: 5

epoch:1	Acc:0.8944310897435898	loss_epoch:0.41509777307510376	lr:0.0009992687385740402
epoch:9	Acc:0.9389022435897436	loss_epoch:0.3735170066356659	lr:0.00016000000000000004
epoch:20	Acc:0.9384348290598291	loss_epoch:0.37389299273490906	lr:2.5600000000000006e-05
Test Loss:  0.38,  Test Acc: 93.47% # select_indegree_num: 36

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_DanMu_2022031702_34gpu1_20ep64bs.log 2>&1 &
[2] 1415265, 
a, added a graph attention mechanism, it first samples a fixed-size neighborhood of each node, then attends over them by a graph attention.
epoch:1	Acc:0.8747996794871795	loss_epoch:0.4328503906726837	lr:0.0009992687385740402
Test Loss:  0.38,  Test Acc: 93.27%

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_MR_2022031703_34gpu1_20ep64bs.log 2>&1 &
[1] 35328, 
Test Loss:  0.67,  Test Acc: 63.19%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6467    0.5913    0.6178       805
    negative     0.6192    0.6730    0.6450       795
    accuracy                         0.6319      1600
   macro avg     0.6330    0.6321    0.6314      1600
weighted avg     0.6331    0.6319    0.6313      1600

Test Loss:  0.66,  Test Acc: 63.50%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6283    0.6720    0.6495       805
    negative     0.6428    0.5975    0.6193       795
    accuracy                         0.6350      1600
   macro avg     0.6355    0.6348    0.6344      1600
weighted avg     0.6355    0.6350    0.6345      1600

Test Loss:  0.65,  Test Acc: 63.69%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6343    0.6571    0.6455       805
    negative     0.6397    0.6164    0.6278       795
    accuracy                         0.6369      1600
   macro avg     0.6370    0.6367    0.6367      1600
weighted avg     0.6370    0.6369    0.6367      1600

Test Loss:  0.64,  Test Acc: 64.19%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6461    0.6373    0.6417       805
    negative     0.6377    0.6465    0.6421       795
    accuracy                         0.6419      1600
   macro avg     0.6419    0.6419    0.6419      1600
weighted avg     0.6419    0.6419    0.6419      1600


(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_MR_2022031704_34gpu1_20ep64bs.log 2>&1 &
[2] 176374, 
a, Update edge representaion
b, update node representaion with one-layer MLP, a GELU in between;
parameters numbers:10526979
epoch:1	Acc:0.5657552083333334	loss_epoch:0.6890655755996704	lr:0.0009931623931623932
epoch:12	Acc:0.6575520833333334	loss_epoch:0.6328744292259216	lr:0.00016000000000000004
epoch:20	Acc:0.6536458333333334	loss_epoch:0.6323181390762329	lr:2.5600000000000006e-05
Test Loss:  0.63,  Test Acc: 65.12% --> the best version so far I have seen.
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6606    0.6311    0.6455       805
    negative     0.6426    0.6717    0.6568       795
    accuracy                         0.6512      1600
   macro avg     0.6516    0.6514    0.6512      1600
weighted avg     0.6517    0.6512    0.6511      1600

Test Loss:  0.62,  Test Acc: 66.25%--> the best version so far I have seen.
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6634    0.6683    0.6658       805
    negative     0.6616    0.6566    0.6591       795
    accuracy                         0.6625      1600
   macro avg     0.6625    0.6625    0.6625      1600
weighted avg     0.6625    0.6625    0.6625      1600

Test Loss:  0.61,  Test Acc: 67.31% --> embedding size 300, lstm_hidden_size 300
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6632    0.7118    0.6866       805
    negative     0.6848    0.6340    0.6584       795
    accuracy                         0.6731      1600
   macro avg     0.6740    0.6729    0.6725      1600
weighted avg     0.6739    0.6731    0.6726      1600

Test Loss:  0.61,  Test Acc: 68.31% --> embedding size 300, lstm_hidden_size 300, graph path only used, without simplies features:
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6693    0.7317    0.6991       805
    negative     0.7000    0.6340    0.6653       795
    accuracy                         0.6831      1600
   macro avg     0.6847    0.6828    0.6822      1600
weighted avg     0.6846    0.6831    0.6823      1600

Test Loss:  0.61,  Test Acc: 68.81% --> embedding size 300, lstm_hidden_size 300, graph path only used, without simplies features, dropout_fc is set to 0.2;
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6783    0.7230    0.6999       805
    negative     0.6995    0.6528    0.6753       795
    accuracy                         0.6881      1600
   macro avg     0.6889    0.6879    0.6876      1600
weighted avg     0.6888    0.6881    0.6877      1600

Test Loss:  0.61,  Test Acc: 69.00% --> 50 epochs, dropout_fc=0.2,dropout=0.0
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6846    0.7118    0.6979       805
    negative     0.6959    0.6679    0.6816       795
    accuracy                         0.6900      1600
   macro avg     0.6903    0.6899    0.6898      1600
weighted avg     0.6902    0.6900    0.6898      1600

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_MR_2022031705_34gpu1_20ep64bs.log 2>&1 &
[1] 276137
a, adds geometry of topological features in the term graph;
epoch:1	Acc:0.564453125	loss_epoch:0.6794178485870361	lr:0.0009931623931623932
epoch:2	Acc:0.6341145833333334	loss_epoch:0.6415495872497559	lr:0.001
epoch:38	Acc:0.69140625	loss_epoch:0.6099905967712402	lr:2.5600000000000006e-05
epoch:49	Acc:0.6809895833333334	loss_epoch:0.6158068776130676	lr:2.5600000000000006e-05
Test Loss:  0.61,  Test Acc: 69.19% -- the best performance so far
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6835    0.7217    0.7021       805
    negative     0.7013    0.6616    0.6809       795
    accuracy                         0.6919      1600
   macro avg     0.6924    0.6917    0.6915      1600
weighted avg     0.6924    0.6919    0.6916      1600

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_MR_2022031706_34gpu1_20ep64bs.log 2>&1 &
[1] 294428, 
a, uses two diff MLP for in-text edge and outer edge;
Test Loss:  0.61,  Test Acc: 67.94% -- bad idea

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_MR_2022031707_34gpu1_20ep64bs.log 2>&1 &
[2] 298836, 
a, adds nn.LayerNorm in Node & Edge fusion;
Test Loss:  0.61,  Test Acc: 68.75% -- it shows nn.LayerNorm brings postive performance

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_MR_2022031708_34gpu1_20ep64bs.log 2>&1 &
[1] 370739, log_MR_2022031705_34gpu1_20ep64bs.log with nn.LayerNorm in Node & Edge fusion;
epoch:1	Acc:0.5833333333333334	loss_epoch:0.6716348528862	lr:0.0009931623931623932
epoch:8	Acc:0.67578125	loss_epoch:0.623614490032196	lr:0.00016000000000000004
epoch:18	Acc:0.6640625	loss_epoch:0.6301901936531067	lr:2.5600000000000006e-05
Test Loss:   0.6,  Test Acc: 69.88%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.7057    0.6882    0.6969       805
    negative     0.6920    0.7094    0.7006       795
    accuracy                         0.6987      1600
   macro avg     0.6989    0.6988    0.6987      1600
weighted avg     0.6989    0.6987    0.6987      1600

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_MR_2022031709_34gpu1_20ep64bs.log 2>&1 &
[1] 398896,
a, In GraphAttnLayer(), we used the residual connection, with the orginal value and the attention value;
epoch:1	Acc:0.591796875	loss_epoch:0.66954505443573	lr:0.0009931623931623932
epoch:10	Acc:0.68359375	loss_epoch:0.6163252592086792	lr:0.00016000000000000004
epoch:21	Acc:0.6770833333333334	loss_epoch:0.6176464557647705	lr:2.5600000000000006e-05
Test Loss:  0.59,  Test Acc: 71.19% -- -- the best performance so far
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.7082    0.7267    0.7174       805
    negative     0.7158    0.6969    0.7062       795
    accuracy                         0.7119      1600
   macro avg     0.7120    0.7118    0.7118      1600
weighted avg     0.7120    0.7119    0.7118      1600


#消融实验，去掉 topology based neighbors 的特征
Test Loss:   0.6,  Test Acc: 69.75%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6913    0.7205    0.7056       805
    negative     0.7043    0.6742    0.6889       795
    accuracy                         0.6975      1600
   macro avg     0.6978    0.6974    0.6973      1600
weighted avg     0.6978    0.6975    0.6973      1600

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_MR_2022031711_34gpu1_20ep64bs.log 2>&1 &
[1] 505195, based on log_MR_2022031709_34gpu1_20ep64bs.log,
a, select_indegree_num: 50, instead of 36;
Test Loss:   0.6,  Test Acc: 70.38%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6982    0.7242    0.7110       805
    negative     0.7098    0.6830    0.6962       795
    accuracy                         0.7037      1600
   macro avg     0.7040    0.7036    0.7036      1600
weighted avg     0.7040    0.7037    0.7036      1600



(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_MR_2022031710_34gpu1_20ep64bs.log 2>&1 &
[2] 402253,
a, Based on log_MR_2022031709_34gpu1_20ep64bs.log, we add nn.LayerNorm in GraphAttnLayer(); -- bad idea
epoch:1	Acc:0.5944010416666666	loss_epoch:0.6693364381790161	lr:0.0009931623931623932
epoch:10	Acc:0.6927083333333334	loss_epoch:0.6104650497436523	lr:0.00016000000000000004
epoch:21	Acc:0.6751302083333334	loss_epoch:0.6226528882980347	lr:2.5600000000000006e-05
Test Loss:  0.59,  Test Acc: 71.06%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.7311    0.6720    0.7003       805
    negative     0.6930    0.7497    0.7202       795
    accuracy                         0.7106      1600
   macro avg     0.7121    0.7109    0.7103      1600
weighted avg     0.7122    0.7106    0.7102      1600

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_MR_2022031711_34gpu1_20ep64bs.log 2>&1 &
[1] 101614, 
a, randomly sample 50 meta-path based neighbors, select_indegree_num: 50, instead of 36;
Test Loss:  0.59,  Test Acc: 71.94% -- the best performance so far
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.7145    0.7366    0.7254       805
    negative     0.7247    0.7019    0.7131       795
    accuracy                         0.7194      1600
   macro avg     0.7196    0.7193    0.7192      1600
weighted avg     0.7195    0.7194    0.7193      1600
epoch:1	Acc:0.6022135416666666	loss_epoch:0.6663457155227661	lr:0.0009931623931623932
epoch:25	Acc:0.697265625	loss_epoch:0.6057791709899902	lr:2.5600000000000006e-05
epoch:36	Acc:0.6790364583333334	loss_epoch:0.6166293621063232	lr:2.5600000000000006e-05


(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_MR_2022031712_34gpu1_20ep64bs.log 2>&1 &
[2] 5263,
a, select_indegree_num = 50;
b, 去掉自循环的外部节点信息；
Test Loss:  0.59,  Test Acc: 71.44%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.7112    0.7280    0.7195       805
    negative     0.7178    0.7006    0.7091       795
    accuracy                         0.7144      1600
   macro avg     0.7145    0.7143    0.7143      1600
weighted avg     0.7145    0.7144    0.7143      1600
epoch:1	Acc:0.5963541666666666	loss_epoch:0.6665644645690918	lr:0.0009931623931623932
epoch:25	Acc:0.693359375	loss_epoch:0.6088361144065857	lr:2.5600000000000006e-05
epoch:36	Acc:0.6790364583333334	loss_epoch:0.6201754212379456	lr:2.5600000000000006e-05

自循环的外部邻居节点，确实带来了性能提升：
    1.self-loop outer nodes for every node brings positive effection for Acc: 0.5963 -> 0.6022 at the 1th epoch;
    2.Acc increases to 71.94% from 71.44% on the test set;


(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_MR_2022031713_34gpu1_20ep64bs.log 2>&1 &
[2] 16887,
a, 自循环的外部邻居节点，消融实验之自循环的外部邻居节点，指 n_{i}->n_{k}->n_{i},n_{k}是外部节点;
Test Loss:  0.59,  Test Acc: 71.94% -- the best performance
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.7145    0.7366    0.7254       805
    negative     0.7247    0.7019    0.7131       795
    accuracy                         0.7194      1600
   macro avg     0.7196    0.7193    0.7192      1600
weighted avg     0.7195    0.7194    0.7193      1600
epoch:1	    Acc:0.6022135416666666	loss_epoch:0.6663457155227661	lr:0.0009931623931623932
epoch:25	Acc:0.697265625	loss_epoch:0.6057791709899902	lr:2.5600000000000006e-05
epoch:36	Acc:0.6790364583333334	loss_epoch:0.6166293621063232	lr:2.5600000000000006e-05

(vd) pw@lenovo:/data2/pw/pathwalk$ nohup bash start.sh > log_MR_2022031714_34gpu1_20ep64bs.log 2>&1 &
[1] 2464, multihead attention with K=2, learning rate is 0.005;
Test Loss:  0.58,  Test Acc: 69.38%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6834    0.7292    0.7055       805
    negative     0.7058    0.6579    0.6810       795
    accuracy                         0.6937      1600
   macro avg     0.6946    0.6935    0.6933      1600
weighted avg     0.6945    0.6937    0.6933      1600
epoch:1	Acc:0.6126302083333334	train_loss:0.6693543791770935	dev_loss:0.625763475894928	lr:0.004965811965811966
epoch:10	Acc:0.6822916666666666	train_loss:0.3721145987510681	dev_loss:0.5938565731048584	lr:0.0008000000000000001
epoch:21	Acc:0.6731770833333334	train_loss:0.3466472029685974	dev_loss:0.6012923717498779	lr:0.0001280000000000000

pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022031715_34gpu1_20ep64bs.log 2>&1 &
[1] 2555894
a, select_indegree_num: 50
b, multihead num = 8
Test Loss:  0.62,  Test Acc: 66.12%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6625    0.6658    0.6642       805
    negative     0.6599    0.6566    0.6583       795
    accuracy                         0.6613      1600
   macro avg     0.6612    0.6612    0.6612      1600
weighted avg     0.6612    0.6613    0.6612      1600


pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022031716_58gpu1_20ep64bs.log 2>&1 &
[1] 2564388, same as log_MR_2022031715_34gpu1_20ep64bs.log, but ,
  dropout_fc: 0.2
  embedding_size: 512
  has_residual: true
  lstm_hidden_size: 512
  lstm_num_layers: 2
  multi_heads: 4
  select_indegree_num: 60
Test Loss:  0.59,  Test Acc: 69.06%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6881    0.7043    0.6961       805
    negative     0.6933    0.6767    0.6849       795
    accuracy                         0.6906      1600
   macro avg     0.6907    0.6905    0.6905      1600
weighted avg     0.6907    0.6906    0.6906      1600

pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022031717_58gpu0_20ep64bs.log 2>&1 &
[1] 2625889, 
  select_indegree_num: 60
  dropout: 0.0
  dropout_fc: 0.3
  embedding_size: 300
  has_residual: true
  lstm_hidden_size: 300
  lstm_num_layers: 2
  multi_heads: 8
Test Loss:  0.57,  Test Acc: 70.38%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.7006    0.7180    0.7092       805
    negative     0.7071    0.6893    0.6981       795
    accuracy                         0.7037      1600
   macro avg     0.7039    0.7037    0.7036      1600
weighted avg     0.7038    0.7037    0.7037      1600


pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022031718_58gpu2_20ep64bs.log 2>&1 &
[1] 2634208, 
  select_indegree_num: 50
  dropout: 0.0
  dropout_fc: 0.6
  embedding_size: 300
  has_residual: true
  lstm_hidden_size: 300
  lstm_num_layers: 2
  multi_heads: 8
Test Loss:  0.59,  Test Acc: 66.62%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6615    0.6894    0.6752       805
    negative     0.6715    0.6428    0.6568       795
    accuracy                         0.6663      1600
   macro avg     0.6665    0.6661    0.6660      1600
weighted avg     0.6665    0.6663    0.6661      1600

pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022031719_58gpu2_20ep64bs.log 2>&1 &
[2] 2650632,
  dropout_fc: 0.2
  embedding_size: 256
  has_residual: true
  lstm_hidden_size: 256
  lstm_num_layers: 2
  multi_heads: 8
Test Loss:  0.58,  Test Acc: 70.00%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6999    0.7068    0.7033       805
    negative     0.7001    0.6931    0.6966       795
    accuracy                         0.7000      1600
   macro avg     0.7000    0.7000    0.7000      1600
weighted avg     0.7000    0.7000    0.7000      1600

pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022031720_58gpu0_20ep64bs.log 2>&1 &
[2] 2661121,
  select_indegree_num: 50
  dropout: 0.0
  dropout_fc: 0.2
  embedding_size: 256
  has_residual: true
  lstm_hidden_size: 256
  lstm_num_layers: 2
  multi_heads: 16
Test Loss:  0.58,  Test Acc: 69.81%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6912    0.7230    0.7067       805
    negative     0.7058    0.6730    0.6890       795
    accuracy                         0.6981      1600
   macro avg     0.6985    0.6980    0.6979      1600
weighted avg     0.6985    0.6981    0.6979      1600


pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022031721_58gpu0_20ep64bs.log 2>&1 &
[1] 2717193,
Test Loss:  0.58,  Test Acc: 69.56%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6819    0.7404    0.7099       805
    negative     0.7121    0.6503    0.6798       795
    accuracy                         0.6956      1600
   macro avg     0.6970    0.6953    0.6949      1600
weighted avg     0.6969    0.6956    0.6950      1600


pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_DanMu_2022031722_58gpu2_20ep64bs.log 2>&1 &
[1] 2720052,
Test Loss:  0.36,  Test Acc: 94.97%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.9626    0.9605    0.9616      9817
    negative     0.9255    0.9294    0.9274      5183
    accuracy                         0.9497     15000
   macro avg     0.9440    0.9449    0.9445     15000
weighted avg     0.9498    0.9497    0.9498     15000

pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022031723_58gpu0_20ep64bs.log 2>&1 &
[2] 2763983
  select_indegree_num: 50
  dropout: 0.0
  dropout_fc: 0.2
  embedding_size: 300
  has_residual: true
  lstm_hidden_size: 300
  lstm_num_layers: 2
  multi_heads: 4
  txt_bidirectional: true
  batch_size: 64
  eta_min: 0.0001
  initial_lr: 0.005
Test Loss:  0.58,  Test Acc: 70.00%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6932    0.7242    0.7084       805
    negative     0.7075    0.6755    0.6911       795
    accuracy                         0.7000      1600
   macro avg     0.7004    0.6998    0.6998      1600
weighted avg     0.7003    0.7000    0.6998      1600
epoch:1	Acc:0.615234375	train_loss:0.666275680065155	dev_loss:0.6313941478729248	lr:0.004965811965811966
epoch:14	Acc:0.689453125	train_loss:0.3568127751350403	dev_loss:0.5892779231071472	lr:0.0003200000000000001
epoch:25	Acc:0.6842447916666666	train_loss:0.3442736268043518	dev_loss:0.589320957660675	lr:0.00012800000000000002

pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022031724_58gpu2_20ep64bs.log 2>&1 &
[1] 2783087,
  select_indegree_num: 50
  dropout: 0.0
  dropout_fc: 0.2
  embedding_size: 300
  has_residual: true
  lstm_hidden_size: 300
  lstm_num_layers: 2
  multi_heads: 4
  txt_bidirectional: true
solver:
  batch_size: 64
  eta_min: 0.0001
  initial_lr: 0.005
Test Loss:  0.58,  Test Acc: 69.62%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6957    0.7043    0.7000       805
    negative     0.6968    0.6881    0.6924       795
    accuracy                         0.6963      1600
   macro avg     0.6963    0.6962    0.6962      1600
weighted avg     0.6963    0.6963    0.6962      1600
epoch:1	Acc:0.615234375	train_loss:0.6662756204605103	dev_loss:0.6313949823379517	lr:0.004965811965811966
epoch:29	Acc:0.6848958333333334	train_loss:0.33995503187179565	dev_loss:0.5979022979736328	lr:0.00015625
epoch:40	Acc:0.6803385416666666	train_loss:0.33740711212158203	dev_loss:0.603264570236206	lr:3.90625e-05

pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022032601_58gpu2_20ep64bs.log 2>&1 &
[1] 3034035,
Test Loss:  0.58,  Test Acc: 69.38%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.7006    0.6832    0.6918       805
    negative     0.6871    0.7044    0.6957       795
    accuracy                         0.6937      1600
   macro avg     0.6939    0.6938    0.6937      1600
weighted avg     0.6939    0.6937    0.6937      1600
Test Loss:  0.59,  Test Acc: 69.25%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6902    0.7056    0.6978       805
    negative     0.6950    0.6792    0.6870       795
    accuracy                         0.6925      1600
   macro avg     0.6926    0.6924    0.6924      1600
weighted avg     0.6926    0.6925    0.6924      1600

pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022032602_58gpu0_20ep64bs.log 2>&1 &
[1] 3086925,
a, adds nn.LayerNorm in MultiHeadAttn;
Test Loss:  0.59,  Test Acc: 68.69%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.7000    0.6609    0.6799       805
    negative     0.6750    0.7132    0.6936       795
    accuracy                         0.6869      1600
   macro avg     0.6875    0.6870    0.6867      1600
weighted avg     0.6876    0.6869    0.6867      1600


pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022032603_58gpu2_20ep64bs.log 2>&1 &
[1] 3152050,
Test Loss:   0.6,  Test Acc: 68.31%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6858    0.6832    0.6845       805
    negative     0.6805    0.6830    0.6817       795
    accuracy                         0.6831      1600
   macro avg     0.6831    0.6831    0.6831      1600
weighted avg     0.6831    0.6831    0.6831      1600
multi_heads: 2

pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022032604_58gpu2_20ep64bs.log 2>&1 &
[1] 3594755,
multi_heads: 1
Test Loss:  0.59,  Test Acc: 69.25%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6925    0.6994    0.6959       805
    negative     0.6925    0.6855    0.6890       795
    accuracy                         0.6925      1600
   macro avg     0.6925    0.6925    0.6925      1600
weighted avg     0.6925    0.6925    0.6925      1600

pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022032605_58gpu1_20ep64bs.log 2>&1 &
[2] 3597510,
multi_heads: 4
Test Loss:  0.59,  Test Acc: 69.44%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6904    0.7118    0.7009       805
    negative     0.6987    0.6767    0.6875       795
    accuracy                         0.6944      1600
   macro avg     0.6945    0.6943    0.6942      1600
weighted avg     0.6945    0.6944    0.6943      1600

pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022032606_58gpu1_20ep64bs.log 2>&1 &
[3] 3599012,
multi_heads: 6
Test Loss:  0.58,  Test Acc: 70.69%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.7049    0.7180    0.7114       805
    negative     0.7090    0.6956    0.7022       795
    accuracy                         0.7069      1600
   macro avg     0.7069    0.7068    0.7068      1600
weighted avg     0.7069    0.7069    0.7068      1600

pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022032607_58gpu1_20ep64bs.log 2>&1 &
[4] 3602218,
multi_heads: 8
Test Loss:  0.59,  Test Acc: 68.56%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6693    0.7416    0.7036       805
    negative     0.7062    0.6289    0.6653       795
    accuracy                         0.6856      1600
   macro avg     0.6877    0.6853    0.6845      1600
weighted avg     0.6876    0.6856    0.6846      1600


pangwei@gpu3:~/pathwalk$ bash start.sh > log_MR_2022032608_58gpu2_20ep64bs.log 2>&1 &
[1] 4057292,
epoch:1	Acc:0.591796875	train_loss:0.6754343509674072	dev_loss:0.6390583515167236	lr:0.004965811965811966
epoch:6	Acc:0.6940104166666666	train_loss:0.4335694909095764	dev_loss:0.5807591676712036	lr:0.0025
epoch:10	Acc:0.6979166666666666	train_loss:0.37710168957710266	dev_loss:0.5816542506217957	lr:0.00125
epoch:15	Acc:0.69921875	train_loss:0.3553721308708191	dev_loss:0.5808966159820557	lr:0.000625
epoch:16	Acc:0.7044270833333334	train_loss:0.353217214345932	dev_loss:0.5771855711936951	lr:0.000625
epoch:27	Acc:0.693359375	train_loss:0.34040921926498413	dev_loss:0.5819498896598816	lr:0.00015625
Test Loss:  0.57,  Test Acc: 70.94%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.7104    0.7130    0.7117       805
    negative     0.7083    0.7057    0.7070       795
    accuracy                         0.7094      1600
   macro avg     0.7094    0.7094    0.7094      1600
weighted avg     0.7094    0.7094    0.7094      1600

pangwei@gpu3:~/pathwalk$ bash start.sh > log_MR_2022032701_58gpu0_20ep64bs.log 2>&1 &
[1] 521761,
Test Loss:  0.58,  Test Acc: 70.50%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6856    0.7640    0.7227       805
    negative     0.7297    0.6453    0.6849       795
    accuracy                         0.7050      1600
   macro avg     0.7077    0.7046    0.7038      1600
weighted avg     0.7075    0.7050    0.7039      1600


pangwei@gpu3:~/pathwalk$ bash start.sh > log_MR_2022032702_58gpu0_20ep64bs.log 2>&1 &
[1] 590501
Test Loss:  0.59,  Test Acc: 68.50%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6720    0.7304    0.7000       805
    negative     0.7007    0.6390    0.6684       795
    accuracy                         0.6850      1600
   macro avg     0.6863    0.6847    0.6842      1600
weighted avg     0.6863    0.6850    0.6843      1600

pangwei@gpu3:~/pathwalk$ bash start.sh > log_MR_2022032703_58gpu0_20ep64bs.log 2>&1 &
[1] 766742
  select_indegree_num: 50
  dropout: 0.0
  dropout_fc: 0.2
  embedding_size: 300
  has_residual: true
  lstm_hidden_size: 300
  lstm_num_layers: 2
  multi_heads: 1
  txt_bidirectional: true
  batch_size: 64
  initial_lr: 0.005                         ./output/2022032703
Test Loss:  0.57,  Test Acc: 71.62%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.7034    0.7540    0.7278       805
    negative     0.7313    0.6780    0.7037       795
    accuracy                         0.7162      1600
   macro avg     0.7174    0.7160    0.7157      1600
weighted avg     0.7173    0.7163    0.7158      1600

pangwei@gpu3:~/pathwalk$ bash start.sh > log_MR_2022032704_58gpu0_20ep64bs.log 2>&1 &
[1] 1715333,
Test Loss:  0.57,  Test Acc: 71.50%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.7105    0.7317    0.7209       805
    negative     0.7198    0.6981    0.7088       795
    accuracy                         0.7150      1600
   macro avg     0.7152    0.7149    0.7149      1600
weighted avg     0.7151    0.7150    0.7149      1600
epoch:1	Acc:0.6243489583333334	train_loss:0.6700989603996277	dev_loss:0.6239962577819824	lr:0.004965811965811966
epoch:16	Acc:0.708984375	train_loss:0.34084028005599976	dev_loss:0.5735126733779907	lr:0.000625
epoch:33	Acc:0.705078125	train_loss:0.33279260993003845	dev_loss:0.5769065022468567	lr:7.8125e-05
epoch:34	Acc:0.701171875	train_loss:0.33324018120765686	dev_loss:0.5785343647003174	lr:7.8125e-05
epoch:35	Acc:0.70703125	train_loss:0.33224374055862427	dev_loss:0.5762240290641785	lr:7.8125e-05
epoch:36	Acc:0.701171875	train_loss:0.3331429958343506	dev_loss:0.5790740251541138	lr:7.8125e-05
epoch:37	Acc:0.7044270833333334	train_loss:0.332303911447525	dev_loss:0.5770923495292664	lr:7.8125e-05
pangwei@gpu3:~/pathwalk$ bash start.sh > log_MR_2022032705_58gpu0_20ep64bs.log 2>&1 &
[1] 1035478, same as log_MR_2022032704_58gpu0_20ep64bs.log, reproducing the prev best reslut.
Test Loss:  0.57,  Test Acc: 71.31%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.7363    0.6696    0.7014       805
    negative     0.6935    0.7572    0.7240       795
    accuracy                         0.7131      1600
   macro avg     0.7149    0.7134    0.7127      1600
weighted avg     0.7151    0.7131    0.7126      1600
[1] 1102618


pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022033101_58gpu1_20ep64bs.log 2>&1 &
[1] 1135746,
Test Loss:  0.58,  Test Acc: 69.12%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.7283    0.6161    0.6676       805
    negative     0.6638    0.7673    0.7118       795
    accuracy                         0.6913      1600
   macro avg     0.6961    0.6917    0.6897      1600
weighted avg     0.6963    0.6913    0.6895      1600


pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022040101_58gpu0_20ep64bs.log 2>&1 &
[2] 2058765, same as 34 log_MR_2022033101_34gpu0_20ep64bs.log, Test Loss:  0.56,  Test Acc: 72.56%;
Test Loss:  0.56,  Test Acc: 72.06%, ./output/2022040101
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.7146    0.7404    0.7273       805
    negative     0.7272    0.7006    0.7136       795
    accuracy                         0.7206      1600
   macro avg     0.7209    0.7205    0.7205      1600
weighted avg     0.7209    0.7206    0.7205      1600
epoch:1	Acc:0.626953125	train_loss:0.6704772710800171	dev_loss:0.6235613822937012	lr:0.004965811965811966
epoch:19	Acc:0.7102864583333334	train_loss:0.33621925115585327	dev_loss:0.5734320878982544	lr:0.0003125
epoch:30	Acc:0.7044270833333334	train_loss:0.3318573236465454	dev_loss:0.578553318977356	lr:0.0003125

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022040102_58gpu2_20ep64bs.log 2>&1 &
[1] 2982889, same as 34 log_MR_2022033103_34gpu1_20ep64bs.log,
Test Loss:  0.58,  Test Acc: 69.50%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6917    0.7106    0.7010       805
    negative     0.6986    0.6792    0.6888       795
    accuracy                         0.6950      1600
   macro avg     0.6951    0.6949    0.6949      1600
weighted avg     0.6951    0.6950    0.6949      1600
epoch:1	Acc:0.5755208333333334	train_loss:0.6654674410820007	dev_loss:0.6624435186386108	lr:0.004965811965811966
epoch:25	Acc:0.7115885416666666	train_loss:0.33899253606796265	dev_loss:0.5730056166648865	lr:0.0003125
epoch:36	Acc:0.70703125	train_loss:0.33503368496894836	dev_loss:0.5760834813117981	lr:0.00015625


(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022040103_58gpu2_20ep64bs.log 2>&1 &
[1] 3626477, same as 34 log_MR_2022033103_34gpu1_20ep64bs.log
Test Loss:  0.56,  Test Acc: 71.75%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.7166    0.7255    0.7210       805
    negative     0.7185    0.7094    0.7139       795
    accuracy                         0.7175      1600
   macro avg     0.7175    0.7174    0.7175      1600
weighted avg     0.7175    0.7175    0.7175      1600

pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022040104_58gpu2_20ep64bs.log 2>&1 &
[1] 3751296, same as log_MR_2022040103_58gpu2_20ep64bs.log
Test Loss:  0.57,  Test Acc: 70.31%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6801    0.7739    0.7240       805
    negative     0.7339    0.6314    0.6788       795
    accuracy                         0.7031      1600
   macro avg     0.7070    0.7027    0.7014      1600
weighted avg     0.7069    0.7031    0.7016      1600


(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022040105_58gpu2_20ep64bs.log 2>&1 &
[1] 3939302, full model + BiLSTM with dropout 0.2;
Test Loss:  0.56,  Test Acc: 71.56%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.7297    0.6907    0.7096       805
    negative     0.7029    0.7409    0.7214       795
    accuracy                         0.7156      1600
   macro avg     0.7163    0.7158    0.7155      1600
weighted avg     0.7163    0.7156    0.7155      1600

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022040106_58gpu2_20ep64bs.log 2>&1 &
[1] 3983153, same as log_MR_2022040105_58gpu2_20ep64bs.log,
a, learning rate 0.001, instead of 0.005;
Test Loss:  0.56,  Test Acc: 71.50%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6994    0.7602    0.7286       805
    negative     0.7338    0.6692    0.7000       795
    accuracy                         0.7150      1600
   macro avg     0.7166    0.7147    0.7143      1600
weighted avg     0.7165    0.7150    0.7144      1600


(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022040107_58gpu2_20ep64bs.log 2>&1 &
[2] 3987719, same as log_MR_2022040105_58gpu2_20ep64bs.log,
a, learning rate 0.001, instead of 0.005;
b, cancels lr_scheduler.LambdaLR;
Test Loss:  0.55,  Test Acc: 73.25%    -- the best accuracy so far. ./output/2022040107
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.7130    0.7839    0.7467       805
    negative     0.7566    0.6805    0.7166       795
    accuracy                         0.7325      1600
   macro avg     0.7348    0.7322    0.7317      1600
weighted avg     0.7347    0.7325    0.7317      1600
epoch:1	Acc:0.6276041666666666	train_loss:0.6688777804374695	dev_loss:0.619897723197937	lr:0.001
epoch:19	Acc:0.7063802083333334	train_loss:0.34460020065307617	dev_loss:0.5728926062583923	lr:0.001
epoch:30	Acc:0.703125	train_loss:0.33528631925582886	dev_loss:0.5800673961639404	lr:0.001

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022040108_58gpu2_20ep64bs.log 2>&1 &
[1] 4171951.
Test Loss:  0.57,  Test Acc: 71.00%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.7052    0.7280    0.7164       805
    negative     0.7152    0.6918    0.7033       795
    accuracy                         0.7100      1600
   macro avg     0.7102    0.7099    0.7099      1600
weighted avg     0.7102    0.7100    0.7099      1600

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022040109_58gpu2_20ep64bs.log 2>&1 &
[1] 92171,
Test Loss:  0.57,  Test Acc: 71.50%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6920    0.7814    0.7340       805
    negative     0.7453    0.6478    0.6931       795
    accuracy                         0.7150      1600
   macro avg     0.7186    0.7146    0.7135      1600
weighted avg     0.7185    0.7150    0.7137      1600

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022040110_58gpu2_20ep64bs.log 2>&1 &
[1] 450048,
Test Loss:  0.56,  Test Acc: 72.50%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6986    0.7975    0.7448       805
    negative     0.7606    0.6516    0.7019       795
    accuracy                         0.7250      1600
   macro avg     0.7296    0.7245    0.7233      1600
weighted avg     0.7294    0.7250    0.7235      1600

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MR_2022040111_58gpu1_20ep64bs.log 2>&1 &
[1] 453786,
Test Loss:  0.58,  Test Acc: 69.62%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
    positive     0.6901    0.7193    0.7044       805
    negative     0.7030    0.6730    0.6877       795
    accuracy                         0.6963      1600
   macro avg     0.6966    0.6961    0.6960      1600
weighted avg     0.6965    0.6963    0.6961      1600

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_CR_2022040622_58gpu2_50ep64bs.log 2>&1 &
[2] 3452835


(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_MPQA_2022040621_58gpu2_50ep64bs.log 2>&1 &
[2] 3437083
Test Loss:  0.48,  Test Acc: 82.77%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support

           1     0.8480    0.9218    0.8834      1126
           2     0.7596    0.5991    0.6699       464

    accuracy                         0.8277      1590
   macro avg     0.8038    0.7605    0.7766      1590
weighted avg     0.8222    0.8277    0.8211      1590

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_CR_2022040630_58gpu2_50ep64bs.log 2>&1 &
[3] 3455065

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_SST1_2022040623_58gpu1_50ep64bs.log 2>&1 &
[4] 3443209,

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_PathWalk_2022051201_58_DanMu.log 2>&1 &
[1] 4074509

### Ablation Study, 20220517
(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_PathWalk_2022051706_58_DanMu_embedsize16.log 2>&1 &
[2] 402186
Test Loss:  0.37,  Test Acc: 94.19%

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_PathWalk_2022051705_58_DanMu_embedsize32.log 2>&1 &
[1] 397266
Test Loss:  0.36,  Test Acc: 95.30%

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_PathWalk_2022051702_58_DanMu_embedsize64.log 2>&1 &
[1] 234280
Test Loss:  0.35,  Test Acc: 95.99%

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_PathWalk_2022051703_58_DanMu_embedsize128.log  2>&1 &
[2] 238032
Test Loss:  0.35,  Test Acc: 96.27%

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_PathWalk_2022051704_58_DanMu_embedsize256.log 2>&1 &
[3] 248204,
Test Loss:  0.35,  Test Acc: 96.37%

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_PathWalk_2022051801_58_DanMu_K0.log 2>&1 &
[2] 1075798
Test Loss:  0.35,  Test Acc: 96.52%

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_PathWalk_2022051807_58_gpu0_DanMu_K5.log 2>&1 &
[4] 21872
Test Loss:  0.35,  Test Acc: 96.47%

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_PathWalk_2022051810_58_gpu1_DanMu_embedsize256LSTM512.log 2>&1 &
[5] 24855
Test Loss:  0.35,  Test Acc: 96.45%

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_PathWalk_2022051811_58_gpu0_DanMu_embedsize384.log 2>&1 &
[1] 37549
Test Loss:  0.35,  Test Acc: 96.64%

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_PathWalk_2022051811_58_gpu1_DanMu_embedsize300.log 2>&1 &
[1] 293948
Test Loss:  0.35,  Test Acc: 96.53%

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_PathWalk_2022051901_58_gpu1_DanMu_embedsize512LSTM512.log 2>&1 &
[1] 759247
Test Loss:  0.35,  Test Acc: 96.47%

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_PathWalk_2022051902_58_gpu2_DanMu_NoEdge.log 2>&1 &
[2] 766063
Test Loss:  0.35,  Test Acc: 95.85%

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_PathWalk_2022052001_58_gpu1_DanMu_embedsize512LSTM300.log 2>&1 &
[3] 2123278


### embedding size: 512, LSTM hidden size: 300
log_PathWalk_2022052002_58_gpu2_DanMu_select_indegree_num_10.log
Test Loss:  0.35,  Test Acc: 96.56%
log_PathWalk_2022052002_58_gpu2_DanMu_select_indegree_num_15.log
Test Loss:  0.35,  Test Acc: 96.68%
log_PathWalk_2022052002_58_gpu2_DanMu_select_indegree_num_20.log
log_PathWalk_2022052002_58_gpu2_DanMu_select_indegree_num_5.log
Test Loss:  0.35,  Test Acc: 96.55%

### embedding size: 300, LSTM hidden size: 300
log_PathWalk_2022052101_58_gpu1_DanMu_300d_select_indegree_num_5.log
Test Loss:  0.35,  Test Acc: 96.53%

log_PathWalk_2022052101_58_gpu1_DanMu_300d_select_indegree_num_10.log
Test Loss:  0.35,  Test Acc: 96.44%

log_PathWalk_2022052101_58_gpu1_DanMu_300d_select_indegree_num_15.log
Test Loss:  0.35,  Test Acc: 96.52%

log_PathWalk_2022052101_58_gpu1_DanMu_300d_select_indegree_num_20.log
Test Loss:  0.35,  Test Acc: 96.59%

log_PathWalk_2022052101_58_gpu1_DanMu_300d_select_indegree_num_25.log
Test Loss:  0.35,  Test Acc: 96.42%

log_PathWalk_2022052101_58_gpu1_DanMu_300d_select_indegree_num_30.log
Test Loss:  0.35,  Test Acc: 96.50%

log_PathWalk_2022052101_58_gpu1_DanMu_300d_select_indegree_num_35.log
Test Loss:  0.35,  Test Acc: 96.54%

log_PathWalk_2022052101_58_gpu1_DanMu_300d_select_indegree_num_40.log
Test Loss:  0.35,  Test Acc: 96.44%
(vd) pangwei@gpu3:~/pathwalk$ tail -n 15 log_PathWalk_2022052101_58_gpu1_DanMu_300d2_select_indegree_num_40.log 
Test Loss:  0.35,  Test Acc: 96.45%

log_PathWalk_2022052101_58_gpu1_DanMu_300d_select_indegree_num_45.log
Test Loss:  0.35,  Test Acc: 96.68%
log_PathWalk_2022052101_58_gpu1_DanMu_300d2_select_indegree_num_45.log
Test Loss:  0.35,  Test Acc: 96.47%
(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_PathWalk_2022052701_58_gpu1_DanMu_300d3_select_indegree_num_45.log 2>&1 &
[1] 204181, 50 epoches
(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_PathWalk_2022052702_58_gpu1_DanMu_300d3_select_indegree_num_45.log 2>&1 &
[2] 207396, 100 epoches
Test Loss:  0.35,  Test Acc: 96.66%


log_PathWalk_2022052101_58_gpu1_DanMu_300d_select_indegree_num_50.log
Test Loss:  0.35,  Test Acc: 96.23%
log_PathWalk_2022052101_58_gpu1_DanMu_300d2_select_indegree_num_50.log
Test Loss:  0.35,  Test Acc: 96.52%
(vd) pangwei@gpu3:~/pathwalk$ tail -n 20 log_PathWalk_2022052101_58_gpu1_DanMu_300d3_select_indegree_num_50.log
100%|██████████| 469/469 [00:37<00:00, 12.59it/s]
Test Loss:  0.35,  Test Acc: 96.52%
(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_PathWalk_2022052703_58_gpu1_DanMu_300d_select_indegree_num_50.log 2>&1 &
[3] 209211, 100 epoches
Test Loss:  0.35,  Test Acc: 96.66%

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > s.log 2>&1 &
[4] 233821

log_PathWalk_2022052101_58_gpu1_DanMu_300d_select_indegree_num_55.log
Test Loss:  0.35,  Test Acc: 96.42%

log_PathWalk_2022052101_58_gpu1_DanMu_300d_select_indegree_num_60.log
Test Loss:  0.35,  Test Acc: 96.42%

log_PathWalk_2022052101_58_gpu1_DanMu_300d_select_indegree_num_65.log
Test Loss:  0.35,  Test Acc: 96.43%

log_PathWalk_2022052101_58_gpu1_DanMu_300d_select_indegree_num_70.log
Test Loss:  0.35,  Test Acc: 96.57%

log_PathWalk_2022052101_58_gpu1_DanMu_300d_select_indegree_num_75.log
Test Loss:  0.35,  Test Acc: 96.43%

log_PathWalk_2022052101_58_gpu1_DanMu_300d_select_indegree_num_80.log
Test Loss:  0.35,  Test Acc: 96.63%


## test \alpha^(2)
(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_PathWalk_2022052601_58_gpu1_DanMu_ablation_alpha2.log 2>&1 &
[2] 3992856
Test Loss:  0.35,  Test Acc: 96.53%
(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_PathWalk_2022052604_58_gpu1_DanMu_ablation_alpha2.log 2>&1 &
[5] 3999219
Test Loss:  0.35,  Test Acc: 96.41%

(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_PathWalk_2022052602_58_gpu0_DanMu_ablation_alpha3.log 2>&1 &
[3] 3995640
Test Loss:  0.35,  Test Acc: 96.03%
(vd) pangwei@gpu3:~/pathwalk$ nohup bash start.sh > log_PathWalk_2022052603_58_gpu0_DanMu_ablation_alpha3.log 2>&1 &
[4] 3997024
Test Loss:  0.35,  Test Acc: 96.23%

## test on max_sequence_length 
(vd) pangwei@gpu3:~/pathwalk$ nohup bash start2.sh > s.log 2>&1 &
[5] 258195
log_PathWalk_2022052705_58_gpu1_DanMu_300d_max_sequence_length_10.log:5311:Test Loss:  0.36,  Test Acc: 95.57%
log_PathWalk_2022052705_58_gpu1_DanMu_300d_max_sequence_length_15.log:5321:Test Loss:  0.36,  Test Acc: 95.49%
log_PathWalk_2022052705_58_gpu1_DanMu_300d_max_sequence_length_20.log:5311:Test Loss:  0.36,  Test Acc: 95.41%
log_PathWalk_2022052705_58_gpu1_DanMu_300d_max_sequence_length_5.log:5317:Test Loss:  0.36,  Test Acc: 95.56%


## K on the CVQD dataset
(vd) pangwei@gpu3:~/pathwalk$ grep -n "Test Acc:" log_PathWalk_2022052202_58_gpu1_CVQD_300d_select_indegree_num_*.log
log_PathWalk_2022052202_58_gpu1_CVQD_300d_select_indegree_num_10.log
Test Loss:   2.8,  Test Acc: 55.45%
log_PathWalk_2022052202_58_gpu1_CVQD_300d_select_indegree_num_15.log
Test Loss:   2.8,  Test Acc: 52.49%
log_PathWalk_2022052202_58_gpu1_CVQD_300d_select_indegree_num_20.log
Test Loss:   2.8,  Test Acc: 55.60%
log_PathWalk_2022052202_58_gpu1_CVQD_300d_select_indegree_num_25.log
Test Loss:   2.8,  Test Acc: 56.34%
log_PathWalk_2022052202_58_gpu1_CVQD_300d_select_indegree_num_30.log
Test Loss:   2.8,  Test Acc: 49.64%
log_PathWalk_2022052202_58_gpu1_CVQD_300d_select_indegree_num_5.log
Test Loss:   2.8,  Test Acc: 55.57%


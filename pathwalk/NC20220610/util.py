import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from torch.nn import functional as F

def evaluate(model,dataloader,iterations,class_list,device,is_test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    for i, batch in tqdm(enumerate(dataloader),total=iterations):
        for key in batch:batch[key] = batch[key].to(device)
        with torch.no_grad():
            y = batch["y"]#[batch_size]
            y_hat = model(batch)#[batch_size,num_class]
            loss = F.cross_entropy(y_hat,y)
            loss_total += loss
            y = y.data.cpu().numpy()
            predic = torch.max(y_hat.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all,y)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    loss_epoch = loss_total / iterations
    if is_test:
        report = metrics.classification_report(labels_all, predict_all, target_names=class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc,loss_epoch,report,confusion
    return acc,loss_epoch


def test(save_path,model,test_dataloader,test_iterations,class_list,device):
    # test
    print("loading model parameters from {} ...".format(save_path))
    model_CKPT = torch.load(save_path)
    model.load_state_dict(model_CKPT['model'])
    model.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate(
        model,test_dataloader,test_iterations,class_list,device, is_test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)


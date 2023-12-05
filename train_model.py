from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torchvision
import time
import copy
from evaluate import fx_calc_map_label
import torch.nn.functional as F
from evaluate import fx_calc_map_label
import numpy as np
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    return Sim

def cal_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2, alpha):

    # Compute pairwise cosine similarity scores between embeddings
    cos = lambda x, y: x.mm(y.t()) / ((x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) / 2.
    ii_sim = cos(view1_feature, view1_feature)
    it_sim = cos(view1_feature, view2_feature)

    temperature = 0.5
    epsilon = 0.9
    # Calculate similarity scores with temperature scaling
    ii_sim = ii_sim * temperature
    it_sim = it_sim * temperature
    
    # Calculate numerator of the softmax
    ii_exp = torch.exp(ii_sim)
    it_exp = torch.exp(it_sim)
    numerator = torch.diag(it_exp)
    # Calculate denominator of the softmax
    denominator = it_exp.sum(dim=1)
    
    # Calculate softmax probabilities
    softmax_probs = numerator / denominator
    
    # Compute negative log probability for positive pairs (diagonal elements)
    soft_loss = -torch.log(softmax_probs)
    
    # Take the mean of the soft loss across samples
    scl = soft_loss.mean()

    # print(view1_feature.shape)
    # print(view1_predict.shape)

    # num_classes1 = labels_1.size(1)
    # smoothed_targets1 = torch.full_like(labels_1, epsilon / (num_classes1 - 1))
    # smoothed_targets1.scatter_(1, labels_1, 1 - epsilon)
    # sce_image = (-torch.sum(smoothed_targets1 * F.log_softmax(view1_predict, dim=1), dim=1)).mean()
    # num_classes2 = labels_2.size(1)
    # smoothed_targets2 = torch.full_like(labels_2, epsilon / (num_classes2 - 1))
    # smoothed_targets2.scatter_(1, labels_2, 1 - epsilon)
    # sce_text = -torch.sum(smoothed_targets2 * F.log_softmax(view2_predict, dim=1), dim=1).mean()
    # sce = sce_image + sce_text

    c_i = view1_predict.size()[-1]
    log_pred_i = torch.log_softmax(view1_predict, dim=-1)
    loss_i = -log_pred_i.sum(dim=-1)
    loss_i = loss_i.mean()
    sce_i = loss_i * epsilon / c_i + (1-epsilon)*nn.CrossEntropyLoss()(view1_predict, labels_1.float())

    c_t = view2_predict.size()[-1]
    log_pred_t = torch.log_softmax(view2_predict, dim=-1)
    loss_t = -log_pred_t.sum(dim=-1)
    loss_t = loss_t.mean()
    sce_t = loss_t * epsilon / c_t + (1-epsilon)*nn.CrossEntropyLoss()(view2_predict, labels_2.float())
    sce = sce_i + sce_t
    # image

    # text

    # print(view1_predict)
    # sce = nn.CrossEntropyLoss(view1_predict, smoothed_targets1) + nn.CrossEntropyLoss(view2_predict, smoothed_targets2)
    # sce = F.binary_cross_entropy_with_logits(view1_predict,smoothed_targets1) + F.binary_cross_entropy_with_logits(view2_predict, smoothed_targets2)
    # print(view1_predict.shape)
    # print(labels_1.shape)
    # sce = F.cross_entropy(view1_predict,labels_1) + F.cross_entropy(view2_predict, labels_2)

    tot_loss = alpha*scl + sce
    return tot_loss



def train_model(model, data_loaders, optimizer, params):
    since = time.time()
    device = params.device
    test_img_acc_history = []
    test_txt_acc_history = []
    epoch_loss_history =[]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(params.epoch):
        print('Epoch {}/{}'.format(epoch+1, params.epoch))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            running_corrects_img = 0
            running_corrects_txt = 0
            # Iterate over data.
            for imgs, txts, labels in data_loaders[phase]:
                # imgs = imgs.to(device)
                # txts = txts.to(device)
                # labels = labels.to(device)
                if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
                    print("Data contains Nan.")

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        labels = labels.cuda()


                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward
                    view1_feature, view2_feature, view1_predict, view2_predict = model(imgs, txts)

                    # loss = calc_loss(view1_feature, view2_feature, view1_predict,
                    #                  view2_predict, labels, labels, alpha, beta)

                    loss = cal_loss(view1_feature, view2_feature, view1_predict,
                                    view2_predict, labels, labels, params.alpha)

                    img_preds = view1_predict
                    txt_preds = view2_predict

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects_img += torch.sum(torch.argmax(img_preds, dim=1) == torch.argmax(labels, dim=1))
                running_corrects_txt += torch.sum(torch.argmax(txt_preds, dim=1) == torch.argmax(labels, dim=1))

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            # epoch_img_acc = running_corrects_img.double() / len(data_loaders[phase].dataset)
            # epoch_txt_acc = running_corrects_txt.double() / len(data_loaders[phase].dataset)
            t_imgs, t_txts, t_labels = [], [], []
            with torch.no_grad():
                for imgs, txts, labels in data_loaders['test']:
                    if torch.cuda.is_available():
                            imgs = imgs.cuda()
                            txts = txts.cuda()
                            labels = labels.cuda()
                    t_view1_feature, t_view2_feature, _, _ = model(imgs, txts)
                    t_imgs.append(t_view1_feature.cpu().numpy())
                    t_txts.append(t_view2_feature.cpu().numpy())
                    t_labels.append(labels.cpu().numpy())
            t_imgs = np.concatenate(t_imgs)
            t_txts = np.concatenate(t_txts)
            t_labels = np.concatenate(t_labels).argmax(1)
            img2text = fx_calc_map_label(t_imgs, t_txts, t_labels)
            txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)

            print('{} Loss: {:.4f} Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(phase, epoch_loss, img2text, txt2img))

            # deep copy the model
            if phase == 'test' and (img2text + txt2img) / 2. > best_acc:
                best_acc = (img2text + txt2img) / 2.
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                test_img_acc_history.append(img2text)
                test_txt_acc_history.append(txt2img)
                epoch_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best average ACC: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, test_img_acc_history, test_txt_acc_history, epoch_loss_history

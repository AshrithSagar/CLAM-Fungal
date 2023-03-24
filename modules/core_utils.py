import numpy as np
import torch
from modules.utils import *
import os
from modules.dataset_generic import save_splits
from modules.model_mil import MIL_fc, MIL_fc_mc
from modules.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import auc as calc_auc

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
#         print("Y_hat", Y_hat)
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
#             print("B-Log Y_hat", Y_hat)
#             print("B-Log Y_hat.shape", Y_hat.shape)
#             print("B-Log Y.shape", Y.shape)
#             print("B-0", sum(cls_mask))
#             print("B-1", sum(Y_hat[cls_mask] == Y[cls_mask]))
            self.data[label_class]["count"] += sum(cls_mask)
            self.data[label_class]["correct"] += sum(Y_hat[cls_mask] == Y[cls_mask])

#         Y_hat = np.array(Y_hat).astype(int)
#         Y_hat = np.reshape(Y_hat, (16, 2))
#         Y_hat = Y_hat[:, 0]
#         Y = np.array(Y).astype(int)
#         for label_class in np.unique(Y):
#             cls_mask = [Y == label_class]
#             self.data[label_class]["count"] += sum(cls_mask)
#             self.data[label_class]["correct"] += sum(tuple([(Y_hat[cls_mask] == Y[cls_mask]) ]) )

#             print("B count:", self.data[label_class]["count"])
#             print("B correct:", self.data[label_class]["correct"])

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, settings):
    """
        train for a single fold
    """
    print("Settings:", settings)
    print('\nTraining Fold {}!'.format(cur))
    exp_dir = os.path.join(settings["results_dir"], str(settings["exp_code"]) + '_s{}'.format(settings["seed"]))
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    split_dir = os.path.join(exp_dir, 'splits_{}'.format(cur))
    if not os.path.isdir(split_dir):
        os.mkdir(split_dir)

    if settings['log_data']:
        writer_dir = os.path.join(split_dir, "logs")
        if not os.path.isdir(writer_dir):
            os.mkdir(writer_dir)

        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(cur)), annot_create=False)
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if settings['bag_loss'] == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = settings['n_classes'])
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')

    if settings['attention_labels_loss'] == 'ce':
        attention_labels_loss_fn = nn.CrossEntropyLoss()
    elif settings['attention_labels_loss'] == 'bce':
        attention_labels_loss_fn = nn.BCELoss()
    else:
        attention_labels_loss_fn = None

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": settings['dropout'], 'n_classes': settings['n_classes']}
    if settings['model_type'] == 'clam' and settings['subtyping']:
        model_dict.update({'subtyping': True})

    if settings['model_size'] is not None and settings['model_type'] != 'mil':
        model_dict.update({"size_arg": settings['model_size']})

    if settings['model_type'] in ['clam_sb', 'clam_mb']:
        if settings['subtyping']:
            model_dict.update({'subtyping': True})

        if settings['B'] > 0:
            model_dict.update({'k_sample': settings['B']})

        if settings['inst_loss'] == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()

        if settings['model_type'] =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn, attention_labels_loss_fn=attention_labels_loss_fn)
        elif settings['model_type'] == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn, attention_labels_loss_fn=attention_labels_loss_fn)
        else:
            raise NotImplementedError

    else: # settings['model_type == 'mil'
        if settings['n_classes'] > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, settings)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = settings['testing'], weighted = settings['weighted_sample'])
    val_loader = get_split_loader(val_split,  testing = settings['testing'])
    test_loader = get_split_loader(test_split, testing = settings['testing'])
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if settings['early_stopping']:
        early_stopping = EarlyStopping(patience = 50, stop_epoch=200, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(settings['max_epochs']):
        weight_alpha = get_alpha_weight(epoch, settings['T1'], settings['T2'], settings['af'], settings['correction'])
        print("\nWeight alpha", weight_alpha)
        if settings['model_type'] in ['clam_sb', 'clam_mb'] and not settings['no_inst_cluster']:
            train_loop_clam(epoch, model, train_loader, optimizer, settings['n_classes'],
                settings['loss_weights'], writer, loss_fn,
                semi_supervised=settings['semi_supervised'],
                alpha_weight=settings['alpha_weight'], weight_alpha=weight_alpha)
            stop = validate_clam(cur, epoch, model, val_loader, settings['n_classes'],
                settings, early_stopping, writer, loss_fn, settings['results_dir'])

        else:
            train_loop(epoch, model, train_loader, optimizer, settings['n_classes'], writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, settings['n_classes'],
                early_stopping, writer, loss_fn, settings['results_dir'])

        if stop:
            break

    exp_dir = os.path.join(settings["results_dir"], str(settings["exp_code"]) + '_s{}'.format(settings["seed"]))
    split_dir = os.path.join(exp_dir, 'splits_{}'.format(cur))
    if settings['early_stopping']:
        model.load_state_dict(torch.load(os.path.join(split_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(split_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _, cm_val, CM_val, cm_val_disp, fpr_val, tpr_val = summary(model, val_loader, settings['n_classes'])
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger, cm_test, CM_test, cm_test_disp, fpr_test, tpr_test = summary(model, test_loader, settings['n_classes'])
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(settings['n_classes']):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error, cm_val, cm_test, CM_val, CM_test, cm_val_disp, cm_test_disp, fpr_val, tpr_val, fpr_test, tpr_test


def train_loop_clam(epoch, model, loader, optimizer, n_classes, loss_weights, writer = None, loss_fn = None, semi_supervised=False, alpha_weight=False, weight_alpha=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0
    labeled_count = 0. # Labelled images
    train_attention_labels_loss = 0.

    print('\n')
    for batch_idx, (data, label, idx, bool_annot, patch_annot) in enumerate(loader):
        data = data.float()
        model = model.float()
        data, label = data.to(device), label.to(device)
        bool_annot, patch_annot = bool_annot.to(device), patch_annot.to(device)
        idx = idx.to(device)
        # print("data.shape", data.shape)
        # print("Index:", idx.item(), "Label:", label.item(), "bool_annot:", bool_annot, "patch_annot:", patch_annot)
        logits, Y_prob, Y_hat, _, instance_dict, attention_labels_loss = model(data, label=label, alpha_weight=alpha_weight, semi_supervised=semi_supervised, bool_annot=bool_annot, patch_annot=patch_annot, weight_alpha=weight_alpha, instance_eval=True, training=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits.view(1, 2), label)
        loss_value = loss.item()

        loss_pos_L1 = instance_dict['loss_pos_L1']
        loss_neg_L1 = instance_dict['loss_neg_L1']

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value

        total_loss = loss_weights['bag'] * loss + loss_weights['instance'] * instance_loss
        if (loss_neg_L1 is not None) and (loss_pos_L1 is not None):
            total_loss += loss_neg_L1 - loss_pos_L1
        if attention_labels_loss is not None:
            attention_labels_loss_value = loss_weights['attention_labels'] * attention_labels_loss
            total_loss += attention_labels_loss_value
            train_attention_labels_loss += attention_labels_loss_value
            labeled_count += 1
        else:
            attention_labels_loss = -1.

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
#         print("inst_preds", inst_preds.shape)
#         print("inst_labels", inst_labels.shape)
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, attention_labels_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, attention_labels_loss, total_loss.item()) +
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    if labeled_count != 0:
        train_attention_labels_loss /= labeled_count

    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)
        if labeled_count != 0:
            writer.add_scalar('train/attention_labels_loss', train_attention_labels_loss, epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits.view(1, 2), label)
        loss_value = loss.item()

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits.view(1, 2), label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error


    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])

    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')


    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    if early_stopping:
        exp_dir = os.path.join(settings["results_dir"], str(settings["exp_code"]) + '_s{}'.format(settings["seed"]))
        assert exp_dir
        split_dir = os.path.join(exp_dir, 'splits_{}'.format(cur))
        assert split_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(split_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam(cur, epoch, model, loader, n_classes, settings, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            logits, Y_prob, Y_hat, _, instance_dict, _ = model(data, label=label, instance_eval=True, training=False)
            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits.view(1, 2), label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']

            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []

    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)


    if early_stopping:
        exp_dir = os.path.join(settings["results_dir"], str(settings["exp_code"]) + '_s{}'.format(settings["seed"]))
        assert exp_dir
        split_dir = os.path.join(exp_dir, 'splits_{}'.format(cur))
        assert split_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(split_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
        cm = confusion_matrix(all_labels, (all_probs[:, 1] > 0.5))
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        cm_disp = ConfusionMatrixDisplay(cm, display_labels=["nonfungal", "fungal"])
        CM_data = {"labels": all_labels, "preds": all_probs[:, 1]}

    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

        cm = []

    return patient_results, test_error, auc, acc_logger, cm, CM_data, cm_disp, fpr, tpr

def get_alpha_weight(epoch, T1, T2, af, correction):
    is_correction = not (epoch % correction)
    if is_correction:
        print("Correction epoch at epoch", epoch)

    if is_correction or (epoch < T1):
        sup = 1.0
        unsup = 0.0
    elif epoch > T2:
        sup = 1.0
        unsup = af
    else:
        sup = 1.0
        unsup = ((epoch-T1) / (T2-T1))*af

    return sup, unsup

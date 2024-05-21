import torch
from torch import nn
from utils.dataloader import *
from utils.metrics import *
from utils import AdamWScheduleFree, SGDScheduleFree
import torch.optim as optim
import time
import models
import mlflow
import logging
import warnings
import argparse
import os
import itertools
warnings.filterwarnings("ignore", category=UserWarning)


def test_fn(testloader, model, num_classes=10):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    losses = AverageMeter('Loss', ':.4e')
    criterion = nn.CrossEntropyLoss()
    acc1 = [0]
    acc5 = [0]
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            batch_size = inputs.size(0)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if num_classes > 5:
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            else:
                acc1 = accuracy(outputs, labels, topk=(1, ))
                acc1 = acc1[0]
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
            losses.update(loss.item(), batch_size)
        if num_classes > 5:
            logging.info(' @Testing * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {losses.avg:.3f}'.format(top1=top1, top5=top5, losses=losses))
            return top1.avg.cpu(), top5.avg.cpu(), losses
        else:
            logging.info(' @Testing * Acc@1 {top1.avg:.3f} Loss {losses.avg:.3f}'.format(top1=top1, losses=losses))
            return top1.avg.cpu(), losses


def training_fn(trainloader, model, optimizer, num_epochs=100, testloader=None, lr_scheduler=None, print_freq=100, save_best_model=False,
                save_path=None, training_mode="BP", num_classes=10, record_gpu_mem = False):
    criterion = nn.CrossEntropyLoss()

    best_acc = 0

    for epoch in range(num_epochs):
        if hasattr(model, "print_stats"):
            model.print_stats()
        if hasattr(model, "reset_statistics"):
            model.reset_statistics()

        logging.info("\t Epoch " + str(epoch) + "...")
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        # losses_fb = AverageMeter('LossKD', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        if num_classes > 5:
            top5 = AverageMeter('Acc@5', ':6.2f')
            progress = ProgressMeter(
                len(trainloader),
                [batch_time, data_time, losses, top1, top5],
                prefix="Epoch: [{}]".format(epoch))
        else:
            progress = ProgressMeter(
                len(trainloader),
                [batch_time, data_time, losses, top1],
                prefix="Epoch: [{}]".format(epoch))

        model.train()
        if hasattr(model, "optimizer_train"):
            model.optimizer_train()
        if optimizer is not None and (isinstance(optimizer, AdamWScheduleFree) or isinstance(optimizer, SGDScheduleFree)):
            optimizer.train()
        end = time.time()
        len_train = len(trainloader)
        for batch_idx, (inputs, labels) in enumerate(trainloader, 0):
            data_time.update(time.time() - end)

            batch_size = inputs.size(0)
            inputs = inputs.cuda()
            labels = labels.cuda()
            if not training_mode == "BP":
                outputs = model(inputs, labels)
                with torch.no_grad():
                    loss = criterion(outputs, labels)
            else:
                optimizer.zero_grad()
                outputs = model(inputs, labels if model.training_mode in ["DFA", "DRTP"] else None)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            if num_classes > 5:
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5) )
            else:
                acc1 = accuracy(outputs, labels, topk=(1, ))
                acc1 = acc1[0]
            top1.update(acc1[0], batch_size)
            if num_classes > 5:
                top5.update(acc5[0], batch_size)
            losses.update(loss.item(), batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % print_freq == (print_freq - 1):
                progress.display(batch_idx, log=True)

            if batch_idx == 10 and record_gpu_mem:
                return

        mlflow.log_metric("acc_train", top1.avg.cpu(), step=epoch)
        if num_classes > 5:
            mlflow.log_metric("acc_train_top5", top5.avg.cpu(), step=epoch)
        mlflow.log_metric("loss_train", losses.avg, step=epoch)
        if num_classes > 5:
            logging.info(' @Training * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {losses.avg:.3f}'.format(top1=top1, top5=top5, losses=losses))
        else:
            logging.info(' @Training * Acc@1 {top1.avg:.3f} Loss {losses.avg:.3f}'.format(top1=top1, losses=losses))
        if testloader:
            model.train()
            if hasattr(model, "optimizer_eval"):
                model.optimizer_eval()
            if optimizer is not None and (isinstance(optimizer, AdamWScheduleFree) or isinstance(optimizer, SGDScheduleFree)):
                optimizer.eval()
            with torch.no_grad():
                for (inputs, labels) in itertools.islice(train_loader, 50):
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    _ = model(inputs)
                model.eval()
            if num_classes > 5:
                acc, acc5, loss_test = test_fn(testloader, model, num_classes)
            else:
                acc, loss_test = test_fn(testloader, model, num_classes)
            if loss_test.avg > 100:
                raise RuntimeError(f"Test Loss: {loss_test.avg} (>100). The model diverge.")
            mlflow.log_metric("acc_test", acc, step=epoch)
            if num_classes > 5:
                mlflow.log_metric("acc_test_top5", acc5, step=epoch)
            mlflow.log_metric("loss_test", loss_test.avg, step=epoch)
            if acc > best_acc:
                best_acc = acc
                if save_best_model:
                    state = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                    }
                    torch.save(state, save_path + f'/model_best.pth.tar')

        if hasattr(model, "lr_scheduler"):
            if isinstance(model.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                if hasattr(model, "lr_scheduler_step"):
                    model.lr_scheduler_step(loss_test.avg)
                else:
                    model.lr_scheduler.step(loss_test.avg)
            else:
                if hasattr(model, "lr_scheduler_step"):
                    model.lr_scheduler_step(loss_test.avg)
                else:
                    model.lr_scheduler.step()
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(loss_test.avg)
            else:
                lr_scheduler.step()
    mlflow.log_metric("best_acc_test", best_acc, step=epoch)


def args_params():
    parser = argparse.ArgumentParser(
        description='Local Learning Rule for Deep Neural Networks Inspired by Neural Activity Synchronization')
    parser.add_argument('--save-path', type=str, default='./experiments',
                        help='Folder to save the log files')
    parser.add_argument('--dataset', type=str,
                        default="CIFAR10", choices=["CIFAR10", "CIFAR100", "CIFAR100AUG", "TinyIMAGENET",
                                                    "CIFAR10AUG", "CIFAR10BASIC", "IMAGENETTE",
                                                    "IMAGENET", "VWW", "FashionMNIST", "MNIST", "IMAGENETTE_BASIC"],
                        help='Name of the dataset to be used for the experiment')
    parser.add_argument('--dataset-dir', type=str, default="~/Datasets",
                        help='Path to the dataset folder')
    parser.add_argument('--model', type=str,
                        default="LLS_SmallConv",
                        help='Name of the model to be used for the experiment')
    parser.add_argument('--num-epochs', type=int,
                        default=1, help='Number of epochs to train the model during evaluation')
    parser.add_argument('--lr', type=float,
                        default=0.1, help='learning rate weights')
    parser.add_argument('--training-mode', type=str,
                        default="LLS", choices=["BP", "LLS", "LocalLosses", "LLS_M", "LLS_MxM", "LLS_MxM_reduced",
                                                "DFA", "DRTP", "LLS_Random", "LLS_M_Random", "LLS_MxM_Random"],
                        help='Feedback mode')
    parser.add_argument('--waveform', type=str,
                        default="cosine", choices=["cosine", "square"],
                        help='Waveform to be used in LLS')
    parser.add_argument('--batch-size', type=int,
                        default=64, help='Batch size')
    parser.add_argument('--test-batch-size', type=int,
                        default=128, help='Test batch size')
    parser.add_argument('--weight-decay', type=float,
                        default=0, help='weight decay')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='Momentum')
    parser.add_argument('--pool', type=str,
                        default="MAX", choices=["MAX", "AVG"],
                        help='pooling layer')
    parser.add_argument('--optimizer', type=str,
                        default="SGD", choices=["SGD", "Adam", "AdamW", "AdamWSF", "SGDSF"],
                        help='Optimizer')
    parser.add_argument('-m', '--milestone', nargs='+',
                        help='Milestones for learning rate scheduler', default=[50, 80, 120, 180])
    parser.add_argument('--gamma', type=float,
                        help='Gamma for learning rate scheduler', default=0.3)
    parser.add_argument('--print-freq', type=int,
                        default=300, help='Frequency of printing statistics')
    parser.add_argument('--experiment-name', type=str,
                        default="LLS",
                        help='Experiment Name for MLFLOW')
    parser.add_argument('--lr-scheduler', type=str,
                        default="ReduceLROnPlateau",
                        choices=["MultiStepLR", "ReduceLROnPlateau"],
                        help='LR scheduler type')
    parser.add_argument('--save-best-model', action='store_true', help='Save best model')
    parser.add_argument('--classes-per-batch', type=int,
                        default=None, help='Classes per batch')
    parser.add_argument('--patience', type=int,
                        default=1000, help='Patience')
    parser.add_argument('--temperature', type=float,
                        default=0.5, help='Temperature')
    parser.add_argument('--label-smoothing', type=float,
                        default=0.0, help='Label Smoothing')
    parser.add_argument('--dropout', type=float,
                        default=0.0, help='Dropout probability')
    parser.add_argument('--record-gpu-mem', action='store_true',
                        help='Record GPU memory usage for 10 batches')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args_params()
    if args.record_gpu_mem:
        torch.cuda.memory._record_memory_history()
    folder_name = f"{args.model}_{args.dataset}_{args.training_mode}_{args.optimizer}_{torch.randint(low=0, high=100000, size=[1]).item()}"
    args.save_path = args.save_path + "/" + folder_name
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Log initialization
    log_path = args.save_path + "/log.log"
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        filename=log_path,
                        filemode="a")
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info(args)
    logging.info('=> Everything will be saved to {}'.format(log_path))

    experiment_name = args.experiment_name
    lr = args.lr
    save_best_model = args.save_best_model
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    model_name = args.model
    dataset = args.dataset
    training_mode = args.training_mode
    momentum = args.momentum
    patience = args.patience
    temperature = args.temperature
    label_smoothing = args.label_smoothing
    dropout = args.dropout
    waveform = args.waveform
    milestones = []
    for point in args.milestone:
        milestones.append(int(point))
    gamma = args.gamma
    optimizer_name = args.optimizer
    print_freq = args.print_freq
    weight_decay = args.weight_decay
    lr_scheduler = args.lr_scheduler
    classes_per_batch = args.classes_per_batch

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        in_channels = 3
        if dataset == "CIFAR10":
            train_loader, _, test_loader = cifar10_basic_dataloader(batch_size=batch_size, test_batch_size=test_batch_size)
            num_classes = 10
            in_channels = 3
        elif dataset == "CIFAR10AUG":
            train_loader, _, test_loader = cifar10_augmented_dataloader(batch_size=batch_size, test_batch_size=test_batch_size)
            num_classes = 10
            in_channels = 3
        elif dataset == "CIFAR100":
            train_loader, _, test_loader = cifar100_dataloader(batch_size=args.batch_size,
                                                               test_batch_size=args.test_batch_size)
            num_classes = 100
        elif dataset == "CIFAR100AUG":
            train_loader, _, test_loader = cifar100_augmented_dataloader(batch_size=args.batch_size,
                                                                         test_batch_size=args.test_batch_size)
            num_classes = 100
        elif dataset == "TinyIMAGENET":
            train_loader, _, test_loader = tinyimagenet_dataloader(batch_size=args.batch_size,
                                                                   test_batch_size=args.test_batch_size,
                                                                   path_for_dataset=args.dataset_dir)
            num_classes = 200
        elif dataset == "IMAGENETTE":
            train_loader, _, test_loader = imagenette_dataloader(batch_size=args.batch_size,
                                                                 test_batch_size=args.test_batch_size)
            num_classes = 10
        elif dataset == "IMAGENETTE_BASIC":
            train_loader, _, test_loader = imagenette_basic_dataloader(batch_size=args.batch_size,
                                                                 test_batch_size=args.test_batch_size)
            num_classes = 10
        elif dataset == "IMAGENET":
            train_loader, _, test_loader = imagenet_dataloader(batch_size=args.batch_size,
                                                               test_batch_size=args.test_batch_size)
            num_classes = 1000
        elif dataset == "VWW":
            train_loader, _, test_loader = vww_dataloader(batch_size=args.batch_size,
                                                          test_batch_size=args.test_batch_size,
                                                          path_for_dataset=args.dataset_dir)
            num_classes = 2
        elif dataset == "MNIST":
            train_loader, _, test_loader = mnist_dataloader(batch_size=batch_size, test_batch_size=test_batch_size)
            num_classes = 10
            in_channels = 1
        elif dataset == "FashionMNIST":
            train_loader, _, test_loader = fashionmnist_dataloader(batch_size=batch_size, test_batch_size=test_batch_size)
            num_classes = 10
            in_channels = 1
        else:
            raise NotImplementedError(f"{dataset} is not supported.")

        model = models.__dict__[model_name](in_features=in_channels, out_features=128, lr=lr, optimizer=optimizer_name,
                                            training_mode=training_mode, n_classes=num_classes, milestones=milestones,
                                            gamma=gamma, momentum=momentum, weight_decay=weight_decay,
                                            lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                            label_smoothing=label_smoothing, dropout=dropout, waveform=waveform)
        model = model.cuda()

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info("Number of trainable params: {0}".format(total_params))

        mlflow.log_param("training_mode", training_mode)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("model", model_name)
        mlflow.log_param("dataset", dataset)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("optimizer", optimizer_name)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("momentum", momentum)
        mlflow.log_param("classes_per_batch", classes_per_batch)
        mlflow.log_param("gamma", gamma)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("lr_scheduler", lr_scheduler)
        mlflow.log_param("patience", patience)
        mlflow.log_param("temperature", temperature)
        mlflow.log_param("label_smoothing", label_smoothing)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("waveform", waveform)

        if training_mode in ["DFA", "DRTP"]:
            training_mode = "BP"

        if training_mode == "BP":
            if optimizer_name == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            elif optimizer_name == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer_name == "AdamWSF":
                optimizer = AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer_name == "SGDSF":
                optimizer = SGDScheduleFree(model.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                raise NotImplementedError("Unknown optimizer")

            if lr_scheduler == "MultiStepLR":
                lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=gamma, milestones=milestones)
            elif lr_scheduler =="ReduceLROnPlateau":
                lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=patience)
        else:
            optimizer = None
            lr_scheduler = None

        training_fn(train_loader, model, optimizer, num_epochs=num_epochs, lr_scheduler=lr_scheduler,
                    testloader=test_loader, training_mode=training_mode, print_freq=print_freq,
                    save_path=args.save_path, save_best_model=args.save_best_model, num_classes=num_classes,
                    record_gpu_mem=args.record_gpu_mem)
        try:
            mlflow.log_artifact(log_path)
        except:
            pass

        try:
            if save_best_model:
                mlflow.log_artifact(args.save_path+"/model_best.pth.tar")
        except:
            pass

    if args.record_gpu_mem:
        torch.cuda.memory._dump_snapshot("{0}.pickle".format(folder_name))

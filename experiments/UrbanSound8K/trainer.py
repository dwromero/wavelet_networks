# torch
import torch
# built-in
import numpy as np
import copy
import experiments.UrbanSound8K.utils as utils
from experiments.UrbanSound8K.tester import test


def train(model, dataloaders, args, test_loader):
    # Define loss
    if 'M' in args.model:
        criterion = torch.nn.CrossEntropyLoss()
    elif '1DCNN' in args.model:
        criterion = torch.nn.CrossEntropyLoss()
    if args.wavelet_loss:
        args.wl = utils.WaveletLoss(weight_loss=1.)

    # Warm-up step
    if args.warm_up == True:
        # warm-up parameters
        epochs_warm_up = 3
        lr_warm_up = 0.1 * args.lr
        # start warm-up
        print('------- Starting warm-up -------')
        # create optimizer
        if args.optim == 'adam':
            optimizer_warm_up = torch.optim.Adam(model.parameters(), lr=lr_warm_up, weight_decay=args.weight_decay)
        if 'M' in args.model:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_warm_up, step_size=1, gamma=2.0)
        _, va, vl, ta, tl = _train(model, epochs_warm_up, criterion, optimizer_warm_up, dataloaders, args.device, lr_scheduler)
        print('------- Finishing warm-up -------')

    # create optimizer
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Get lr scheduler
    if 'M' in args.model:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    else: lr_scheduler = None

    # train network
    _, va, vl, ta, tl = _train(model, args.epochs, criterion, optimizer, dataloaders, args.device, lr_scheduler, args, test_loader)
    # save model
    torch.save(model.state_dict(), args.path)
    # save history
    history = np.array([va, vl, ta, tl])
    np.save(args.path[:-4] + "_history.npy", history)


def _train(model, epochs, criterion, optimizer, dataloader, device, lr_scheduler, args, test_loader):
    # Accumulate information about the training history
    val_acc_history = []
    train_acc_history = []
    loss_train_history = []
    loss_val_history = []
    # Save best performing weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # iterate over epochs
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 30)
        # Print current learning rate
        for param_group in optimizer.param_groups:
            print('Learning Rate: {}'.format(param_group['lr']))
        print('-' * 30)

        # Each epoch consist of training and validation
        for phase in ['train', 'validation']:
            if phase == 'train': model.train()
            else: model.eval()

            # Accumulate accuracy and loss
            running_loss = 0
            running_corrects = 0
            total = 0
            # iterate over data
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # FwrdPhase:
                    outputs = model(inputs)
                    if args.wavelet_loss:
                        loss = criterion(outputs, labels) + args.wl(model)
                    else:
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    # BwrdPhase:
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)

            # statistics of the epoch
            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            import datetime
            print(datetime.datetime.now())

            # Store results
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                loss_train_history.append(epoch_loss)
            if phase == 'validation':
                val_acc_history.append(epoch_acc)
                loss_val_history.append(epoch_loss)

                # Update step in ReduceOnPlateauScheduler
                if lr_scheduler is not None:
                    if lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                        lr_scheduler.step(epoch_loss)

            # If better validation accuracy, replace best weights
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Test accuracy:')
                # Clean CUDA Memory
                del inputs, outputs, labels
                torch.cuda.empty_cache()
                # Perform test
                test(model, test_loader, device)
        print()

    # Report best results
    print('Best Val Acc: {:.4f}'.format(best_acc))
    # Load best model weights
    model.load_state_dict(best_model_wts)
    # Return model and histories
    return model, val_acc_history, loss_val_history, train_acc_history, loss_train_history










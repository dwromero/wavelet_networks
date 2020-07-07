# torch
import torch
# built-in
import numpy as np
import copy
import datetime
import experiments.MagnaTagATune.utils as utils


def train(model, dataloaders, args):
    # Define loss
    criterion = torch.nn.BCEWithLogitsLoss()

    if args.wavelet_loss:
        args.wl = utils.WaveletLoss(weight_loss=1.)

    # create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    # Get lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2)

    # train network
    _train(model, args.epochs, criterion, optimizer, dataloaders, args.device, lr_scheduler, args)
    # save model
    torch.save(model.state_dict(), args.path)


def _train(model, epochs, criterion, optimizer, dataloader, device, lr_scheduler, args):

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
            epoch_loss = 0.0
            labels_set = []
            predicted_set = []

            # iterate over data
            for i, data in enumerate(dataloader[phase]):
                inputs = data['audio'].to(device)
                labels = data['label'].to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # FwrdPhase:
                    outputs = model(inputs)
                    if args.wavelet_loss:
                        loss = criterion(outputs, labels) + args.wl(model)
                    else:
                        loss = criterion(outputs, labels)
                    # BwrdPhase:
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_loss += loss.item()
                labels_set.append(labels.detach().cpu().numpy())
                predicted_set.append(outputs.detach().cpu().numpy())

                if (i + 1) % 20 == 0:
                    print("Epoch [%d/%d], Iter [%d/%d] loss : %.4f" % (
                    epoch + 1, epochs, i + 1, len(dataloader[phase]), loss.item()))

            # statistics of the epoch
            labels_set = np.concatenate(labels_set, axis=0)
            predicted_set = np.concatenate(predicted_set, axis=0)
            # retrieval
            auc1, ap1 = utils.tagwise_aroc_ap(labels_set, predicted_set)
            # annotation
            auc2, ap2 = utils.itemwise_aroc_ap(labels_set, predicted_set)

            print("Retrieval : AROC = %.3f, AP = %.3f / " % (np.mean(auc1), np.mean(ap1)),
                   "Annotation : AROC = %.3f, AP = %.3f" % (np.mean(auc2), np.mean(ap2)))

            print('Average loss: {:.4f} \n'.format(epoch_loss/len(dataloader[phase])))
            print(datetime.datetime.now())

            # Store results
            if phase == 'validation':
                # Update step in ReduceOnPlateauScheduler
                if lr_scheduler is not None:
                    if lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                        lr_scheduler.step(epoch_loss)

        # Eearly stop
        curr_lr = optimizer.param_groups[0]['lr']
        print('Learning rate : {}'.format(curr_lr))
        if curr_lr < 1e-7:
            print("Early stopping")
            break
        print()

    # Return model and histories
    return model

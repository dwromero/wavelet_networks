import torch
import numpy as np
import experiments.MagnaTagATune.utils as utils


def test(model, test_loader, device):
    print("Testing ...")
    # send model to device
    model.eval()
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()

    # Accumulate accuracy and loss
    epoch_loss = 0.0
    labels_set = []
    predicted_set = []

    with torch.no_grad():
        # Iterate through data
        for data in test_loader:
            inputs = data['audio'].to(device)
            labels = data['label'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            epoch_loss += loss.item()
            labels_set.append(labels.detach().cpu().numpy())
            predicted_set.append(outputs.detach().cpu().numpy())

    # Print results
    labels_set = np.concatenate(labels_set, axis=0)
    predicted_set = np.concatenate(predicted_set, axis=0)
    # retrieval
    auc1, ap1 = utils.tagwise_aroc_ap(labels_set, predicted_set)
    # annotation
    auc2, ap2 = utils.itemwise_aroc_ap(labels_set, predicted_set)

    print("Retrieval : AROC = %.3f, AP = %.3f / " % (np.mean(auc1), np.mean(ap1)),
          "Annotation : AROC = %.3f, AP = %.3f" % (np.mean(auc2), np.mean(ap2)))

    avg_loss = epoch_loss / len(test_loader)
    print('Average loss: {:.4f} \n'.format(epoch_loss/len(test_loader)))
    # Return results
    return avg_loss

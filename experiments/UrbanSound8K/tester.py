import torch


def test(model, test_loader, device):
    # send model to device
    model.eval()
    model.to(device)

    # Summarize results
    lbls = []
    pred = []
    correct = 0
    total = 0

    with torch.no_grad():
        # Iterate through data
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print results
    print('Accuracy of the network on the {} test images: {}'.format(total, (100 * correct / total)))
    # Return results
    return correct/total, lbls, pred
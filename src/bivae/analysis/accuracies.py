'''

Functions to compute the cross/joint accuracies

'''

import torch


def conditional_labels(model,classifier1,classifier2, data, n_data=8, ns=30):
    """ Sample ns from the conditional distribution (for each of the first n_data)
    and compute the labels in this conditional distribution (based on the
    predefined classifiers)"""

    bdata = [d[:n_data] for d in data]
    samples = model._sample_from_conditional(bdata, n=ns)
    cross_samples = [torch.stack(samples[0][1]), torch.stack(samples[1][0])]

    # Compute the labels
    preds2 = classifier2(cross_samples[0].permute(1, 0, 2, 3, 4).resize(n_data * ns, 3, 32, 32))  # 8*n x 10
    labels2 = torch.argmax(preds2, dim=1).reshape(n_data, ns)

    preds1 = classifier1(cross_samples[1].permute(1, 0, 2, 3, 4).resize(n_data * ns, 1, 28, 28))  # 8*n x 10
    labels1 = torch.argmax(preds1, dim=1).reshape(n_data, ns)

    return labels2, labels1


def compute_accuracies(model, classifier1, classifier2, data, classes, n_data=20, ns=100):

    """ Given the data, we sample from the conditional distribution and compute conditional
    accuracies. We also sample from the joint distribution of the model and compute
    joint accuracy"""

    # Compute cross_coherence
    labels2, labels1 = conditional_labels(model,classifier1,classifier2,data, n_data, ns)

    # Create an extended classes array where each original label is replicated ns times
    classes_mul = torch.stack([classes[0][:n_data] for _ in range(ns)]).permute(1,0).cuda()
    acc2 = torch.sum(classes_mul == labels2)/(n_data*ns)
    acc1 = torch.sum(classes_mul == labels1)/(n_data*ns)

    metrics = dict(accuracy1 = acc1, accuracy2 = acc2)
    data = model.generate('', 0, N=100, save=False)
    labels_mnist = torch.argmax(classifier1(data[0]), dim=1)
    labels_svhn = torch.argmax(classifier2(data[1]), dim=1)

    joint_acc = torch.sum(labels_mnist == labels_svhn) / 100
    metrics['joint_coherence'] = joint_acc

    return metrics

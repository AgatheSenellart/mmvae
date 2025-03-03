import torch
from torchvision import datasets, transforms
from medmnist import PathMNIST, BloodMNIST, PneumoniaMNIST

def rand_match_on_idx(l1, idx1, l2, idx2, max_d=10000, dm=10):
    """
    l*: sorted labels
    idx*: indices of sorted labels in original list
    """
    _idx1, _idx2 = [], []
    for l in l1.unique():  # assuming both have same idxs
        l_idx1, l_idx2 = idx1[l1 == l], idx2[l2 == l]
        n = min(l_idx1.size(0), l_idx2.size(0), max_d)
        print(n)
        l_idx1, l_idx2 = l_idx1[:n], l_idx2[:n]
        for _ in range(dm):
            _idx1.append(l_idx1[torch.randperm(n)])
            _idx2.append(l_idx2[torch.randperm(n)])
    return torch.cat(_idx1), torch.cat(_idx2)

if __name__ == '__main__':
    max_d = 10000  # maximum number of datapoints per class
    dm = 3      # data multiplier: random permutations to match

    # get the individual datasets
    tx = transforms.ToTensor()

    for split in ['train', 'test']:
        d1 = PneumoniaMNIST(split=split, transform=tx, download=True)
        d2 = datasets.SVHN('../data', split=split, download=True, transform=tx)
        d2.labels = torch.LongTensor(d2.labels.squeeze().astype(int)) % 10

        l1, li1 = torch.tensor(d1.labels.squeeze()).sort()
        l2, li2 = d2.labels.sort()

        
        print(l1.unique(), l2.unique())
        idx1, idx2 = rand_match_on_idx(l1, li1, l2, li2, max_d=max_d, dm=dm)
        shuffle = torch.randperm(len(idx1))
        print(idx1[:4], idx2[:4])
        print('len {} idx:'.format(split), len(idx1), len(idx2))
        torch.save(idx1[shuffle], '../data/' + split + 'chest_s-idx.pt')
        torch.save(idx2[shuffle], '../data/' + split + 'c_svhn-idx.pt')

    
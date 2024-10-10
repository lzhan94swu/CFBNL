from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, feat_mats, net_index, ind_labels, state_labels):
        self.feat_mats = feat_mats
        self.net_index = net_index
        self.ind_labels = ind_labels
        self.state_labels = state_labels

    def __len__(self):
        return len(self.feat_mats)

    def __getitem__(self, idx):
        network_matrix = self.feat_mats[idx]
        net_index = self.net_index[idx]
        ind_label = self.ind_labels[idx]
        state_label = self.state_labels[idx]
        return network_matrix, net_index, ind_label, state_label
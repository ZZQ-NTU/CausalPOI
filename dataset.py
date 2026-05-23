import torch
import numpy as np
from torch.utils.data import Dataset


class PreprocessedPOIDataset(Dataset):
    def __init__(self, poi_ids, checkin_dict, subgraph_dict, mode='train', checkin_mean=None, checkin_std=None):
        self.samples = []
        self.mode = mode
        self.week_num = 4
        self.checkin_dict = checkin_dict
        self.subgraph_dict = subgraph_dict

        for pid in poi_ids:
            weeks = sorted(checkin_dict.get(pid, {}).keys())
            if len(weeks) < self.week_num:
                continue

            week_seq = weeks[:self.week_num]

            if all((pid, w, t) in subgraph_dict for w in week_seq for t in (0, 1)):
                self.samples.append((pid, week_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, weeks = self.samples[idx]

        graphs_treated, centers_treated = [], []
        graphs_control, centers_control = [], []

        for w in weeks:
            g1, c1 = self.subgraph_dict[(pid, w, 1)]
            g0, c0 = self.subgraph_dict[(pid, w, 0)]

            graphs_treated.append(g1)
            centers_treated.append(c1)
            graphs_control.append(g0)
            centers_control.append(c0)

        label_raw = [self.checkin_dict[pid].get(w, 0) for w in weeks]
        label = torch.tensor([np.log1p(float(c)) for c in label_raw], dtype=torch.float)

        return (
            pid,
            graphs_treated, centers_treated, torch.tensor(1), label,
            graphs_control, centers_control, torch.tensor(0), label
        )


def custom_collate(batch):
    poi_ids = [item[0] for item in batch] * 2

    treated_graphs, treated_indices, treated_flags, treated_labels = [], [], [], []
    control_graphs, control_indices, control_flags, control_labels = [], [], [], []

    for item in batch:
        _, g_t, i_t, t_flag, y_t, g_c, i_c, c_flag, y_c = item

        treated_graphs.append(g_t)
        treated_indices.append(i_t)
        treated_flags.append(t_flag)
        treated_labels.append(y_t)

        control_graphs.append(g_c)
        control_indices.append(i_c)
        control_flags.append(c_flag)
        control_labels.append(y_c)

    g_list = treated_graphs + control_graphs
    idx_list = treated_indices + control_indices
    treatment = torch.cat([
        torch.stack(treated_flags).long(),
        torch.stack(control_flags).long()
    ])
    labels = torch.cat([torch.stack(treated_labels), torch.stack(control_labels)])

    perm = torch.randperm(len(g_list))
    g_list = [g_list[i] for i in perm]
    idx_list = [idx_list[i] for i in perm]
    treatment = treatment[perm]
    labels = labels[perm]
    poi_ids = [poi_ids[i] for i in perm]

    return g_list, idx_list, treatment, labels, poi_ids

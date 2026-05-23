import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Batch
import config
from utils import pe_encoding


class POIGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.gat1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=2,
            concat=False,
            edge_dim=1
        )
        self.gn1 = nn.BatchNorm1d(hidden_channels)
        self.gat2 = GATv2Conv(
            hidden_channels,
            out_channels,
            heads=2,
            concat=False,
            edge_dim=1
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_attr):
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr

        x = self.gat1(x, edge_index, edge_attr)
        x = self.gn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index, edge_attr)

        return x


class GATGRUDragonModel(nn.Module):
    def __init__(self, gnn, dragonnet, poi_dict):
        super().__init__()

        self.gnn = gnn
        self.dragonnet = dragonnet
        self.poi_dict = poi_dict

        self.hidden_dim = 768 * 2 + config.pe_size
        self.gru = nn.GRU(input_size=self.hidden_dim, hidden_size=256, batch_first=True)
        self.layernorm = nn.LayerNorm(self.hidden_dim)

    def forward(self, list_of_graphs, list_of_center_indices, list_of_center_ids):
        flat_graphs = [g for graphs in list_of_graphs for g in graphs]
        batched_graph = Batch.from_data_list(flat_graphs).to(config.device)

        h_all = self.gnn(
            batched_graph.x,
            batched_graph.edge_index,
            batched_graph.edge_attr
        )

        node_embeds = []
        node_offset = 0

        for sample_idx, graphs in enumerate(list_of_graphs):
            pid = list_of_center_ids[sample_idx]

            lat = float(self.poi_dict[pid]['lat'])
            lon = float(self.poi_dict[pid]['lon'])
            pe = pe_encoding([lat, lon], dim=config.pe_size)
            pe_tensor = torch.tensor(pe, dtype=torch.float32, device=config.device)

            week_embeds = []

            for week_graph_idx, g in enumerate(graphs):
                num_nodes = g.num_nodes
                center_idx = list_of_center_indices[sample_idx][week_graph_idx]

                h_graph = h_all[node_offset: node_offset + num_nodes]
                h_target = h_graph[center_idx]

                neighbor_mask = torch.ones(num_nodes, dtype=torch.bool, device=config.device)
                neighbor_mask[center_idx] = False

                if neighbor_mask.sum() > 0:
                    h_neigh = h_graph[neighbor_mask].mean(dim=0)
                else:
                    h_neigh = torch.zeros_like(h_target)

                h_week = torch.cat([h_target, h_neigh, pe_tensor], dim=-1)
                h_week = self.layernorm(h_week)
                week_embeds.append(h_week)

                node_offset += num_nodes

            week_seq = torch.stack(week_embeds).unsqueeze(0)  # [1, W, D]
            _, h_last = self.gru(week_seq)  # [1, 1, 256]

            node_embeds.append(h_last.squeeze(0).squeeze(0))

        final_input = torch.stack(node_embeds)  # [B, 256]
        return self.dragonnet(final_input)


class DragonNetPredictor(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, out_weeks=4):
        super().__init__()
        self.out_weeks = out_weeks

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.treat_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.treat_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.y0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_weeks)
        )

        self.y1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_weeks)
        )

    def forward(self, x):
        h_shared = self.shared(x)
        h_treat = self.treat_encoder(x)

        y0 = self.y0(h_shared)
        y1 = self.y1(h_shared)
        t_logit = self.treat_head(h_treat)

        return y0, y1, t_logit

import os
import glob
import json
import math
import pickle
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from transformers import BertTokenizer, RobertaTokenizer, DistilBertTokenizer
from transformers import BertModel, RobertaModel, DistilBertModel
import torch.nn as nn
import config


class LazySubgraphLoader:
    def __init__(self, subgraph_dir, cache_all=True):
        self.subgraph_dir = subgraph_dir
        self.index = {}
        self.cache = {}

        print("Building index for subgraph parts...")
        part_files = sorted(glob.glob(os.path.join(subgraph_dir, 'subgraphs_part*.pkl')))

        for file in tqdm(part_files, desc="Indexing subgraphs"):
            try:
                with open(file, 'rb') as f:
                    part = pickle.load(f)
                    for key in part.keys():
                        self.index[key] = file

                if cache_all:
                    self.cache[file] = part
            except Exception as e:
                print(f"Failed to index {file}: {e}")

    def __getitem__(self, key):
        if key not in self.index:
            raise KeyError(f"Key {key} not found in subgraphs.")

        file = self.index[key]
        if file not in self.cache:
            with open(file, 'rb') as f:
                self.cache[file] = pickle.load(f)

        return self.cache[file][key]

    def __contains__(self, key):
        return key in self.index


def get_alpha(function_alpha, pid, nid, default=1.0):
    """
    function.json format:
        {
            "pid,nid": alpha_value,
            ...
        }
    """
    key = f"{pid},{nid}"
    return float(function_alpha.get(key, default))


def build_one_graph(args):
    pid, week, poi_dict, neighbour_dict, poi_embeddings, function_alpha, with_center = args

    neighbour_info = neighbour_dict.get(pid, {}).get(week, [])
    if not neighbour_info:
        return None

    all_nodes = [pid] + [nid for _, nid in neighbour_info]
    node_idx = {nid: i for i, nid in enumerate(all_nodes)}
    center = 0

    edge_index, edge_weight, x = [], [], []

    for dist, nid in neighbour_info:
        if nid not in node_idx or pid not in node_idx:
            continue

        spatial_w = math.exp(-(float(dist) ** 2) / (2 * config.sigma ** 2))

        # Treatment graph uses alpha(p,n); control graph removes functional effect.
        if with_center:
            alpha_pn = get_alpha(function_alpha, pid, nid, default=1.0)
        else:
            alpha_pn = 1.0

        w = alpha_pn * spatial_w

        # Keep directed edge as in the current paper/code setting.
        edge_index.append([node_idx[pid], node_idx[nid]])
        edge_weight.append(w)

    if len(edge_index) == 0:
        return None

    try:
        for nid in all_nodes:
            if nid == pid and not with_center:
                feat = torch.zeros(768)
            else:
                feat = poi_embeddings[nid]
            x.append(feat)
    except KeyError:
        print(f"[Embedding missing] Some POI embedding missing for graph {pid}-{week}")
        return None

    data = Data(
        x=torch.stack(x),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_weight, dtype=torch.float)
    )

    treatment = 1 if with_center else 0
    return ((pid, week, treatment), (data, center))


def load_or_compute_poi_embeddings(poi_dict, emb_path):
    if os.path.exists(emb_path):
        poi_embeddings = torch.load(emb_path, map_location='cpu')
        print("Loaded precomputed POI embeddings.")
        return poi_embeddings

    print("Computing POI embeddings on CPU with batch inference...")

    if config.lm == 'bert':
        tokenizer = BertTokenizer.from_pretrained(config.lm_names[config.lm])
        model = BertModel.from_pretrained(config.lm_names[config.lm])
    elif config.lm == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(config.lm_names[config.lm])
        model = RobertaModel.from_pretrained(config.lm_names[config.lm])
    else:
        tokenizer = DistilBertTokenizer.from_pretrained(config.lm_names[config.lm])
        model = DistilBertModel.from_pretrained(config.lm_names[config.lm])

    model.eval()

    projection = nn.Sequential(
        nn.Linear(config.lm_hidden_sizes[config.lm], 768),
        nn.ReLU(),
        nn.Linear(768, 768)
    )

    batch_list = list(poi_dict.items())
    poi_embeddings = {}

    for i in tqdm(range(0, len(batch_list), config.bs), desc="Batch Encoding POIs"):
        batch = batch_list[i:i + config.bs]
        s_list = [
            ' '.join(
                str(meta.get(k, '')).replace('_', ' ')
                for k in ['category', 'name', 'street', 'city', 'state', 'postcode']
            )
            for pid, meta in batch
        ]

        tokens = tokenizer(
            ['[CLS] ' + s + ' [SEP]' for s in s_list],
            return_tensors='pt',
            padding=True,
            truncation=True
        )

        with torch.inference_mode():
            output = model(**tokens).last_hidden_state.mean(1)
            vecs = projection(output)

        for (pid, _), vec in zip(batch, vecs):
            poi_embeddings[pid] = vec.cpu()

    torch.save(poi_embeddings, emb_path)
    print("Saved POI embeddings.")
    return poi_embeddings


def load_or_build_subgraphs(graph_path, poi_dict, neighbour_dict, checkin_dict, all_pois, emb_path, function_alpha):
    subgraph_dir = graph_path.replace('.pkl', '_parts')

    if os.path.exists(graph_path):
        subgraph_dict = LazySubgraphLoader(subgraph_dir)
        print("Loaded precomputed subgraphs.")
        return subgraph_dict

    poi_embeddings = load_or_compute_poi_embeddings(poi_dict, emb_path)

    print("Constructing and saving subgraphs with low-memory single-threading...")
    os.makedirs(subgraph_dir, exist_ok=True)
    subgraph_dict = {}

    task_list = []
    for pid in all_pois:
        weeks = sorted(checkin_dict.get(pid, {}).keys())[:4]
        if len(weeks) < 4:
            continue

        neighbor_counts = [len(neighbour_dict.get(pid, {}).get(w, [])) for w in weeks]

        if all(c > 1 for c in neighbor_counts):
            for week in weeks:
                task_list.append((pid, week, poi_dict, neighbour_dict, poi_embeddings, function_alpha, True))
                task_list.append((pid, week, poi_dict, neighbour_dict, poi_embeddings, function_alpha, False))

    save_every = 10000
    count = 0
    file_count = 0

    try:
        for result in tqdm(map(build_one_graph, task_list), total=len(task_list), desc='Building'):
            if result is not None:
                key, value = result
                subgraph_dict[key] = value
                count += 1

                if count % save_every == 0:
                    part_path = os.path.join(subgraph_dir, f"subgraphs_part{file_count}.pkl")
                    with open(part_path, 'wb') as f:
                        pickle.dump(subgraph_dict, f)
                    subgraph_dict.clear()
                    file_count += 1

        if subgraph_dict:
            part_path = os.path.join(subgraph_dir, f"subgraphs_part{file_count}.pkl")
            with open(part_path, 'wb') as f:
                pickle.dump(subgraph_dict, f)
            print(f"[Final] Saved last {len(subgraph_dict)} subgraphs to {part_path}")
            subgraph_dict.clear()

    except Exception as e:
        print(f"Subgraph construction interrupted: {e}")
        if subgraph_dict:
            part_path = os.path.join(subgraph_dir, f"subgraphs_part{file_count}_crash.pkl")
            with open(part_path, 'wb') as f:
                pickle.dump(subgraph_dict, f)
            print(f"Saved partial subgraphs after crash to {part_path}")

    print("Merging all saved subgraph parts...")

    part_files = sorted(glob.glob(os.path.join(subgraph_dir, 'subgraphs_part*.pkl')))

    total_count = 0
    merged = {}
    for file in tqdm(part_files, desc="Merging subgraph parts"):
        try:
            with open(file, 'rb') as f_in:
                part = pickle.load(f_in)
                merged.update(part)
                total_count += len(part)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    with open(graph_path, 'wb') as f_out:
        pickle.dump(merged, f_out)

    print(f"Merged total of {total_count} subgraphs into final dict.")

    return LazySubgraphLoader(subgraph_dir)

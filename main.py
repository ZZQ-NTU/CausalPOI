import argparse
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import logging

import config
from data_utils import load_json, build_checkin_dict, get_all_weeks
from dataset import PreprocessedPOIDataset, custom_collate
from graph_builder import load_or_build_subgraphs
from models import POIGAT, GATGRUDragonModel, DragonNetPredictor
from train_eval import train_regression_model


logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")


def parse_args():
    parser = argparse.ArgumentParser(description='CausalPOI modular codebase')
    parser.add_argument('-c', '--dataset', type=str, default='Overseas')
    parser.add_argument('--runs', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = args.dataset if args.dataset in ['Midwest', 'Northeast', 'South', 'West'] else 'Overseas'

    data_path = f'dataset/{dataset}/data-0-400.json'
    neighbour_path = f'dataset/{dataset}/neighbours.json'
    poi_path = 'dataset/poi.json'
    emb_path = 'dataset/poi_embeddings.pkl'

    # Use a new graph cache to avoid loading old graphs without alpha(p,n).
    graph_path = f'dataset/{dataset}/pregraphs_functional.pkl'
    function_path = f'dataset/{dataset}/function.json'

    data_dict = load_json(data_path)
    neighbour_dict = load_json(neighbour_path)
    poi_dict = load_json(poi_path)
    function_alpha = load_json(function_path)

    print('Successfully loaded', dataset, 'dataset')

    checkin_dict = build_checkin_dict(data_dict)
    all_weeks = get_all_weeks(data_dict)
    all_pois = list(data_dict.keys())

    subgraph_dict = load_or_build_subgraphs(
        graph_path=graph_path,
        poi_dict=poi_dict,
        neighbour_dict=neighbour_dict,
        checkin_dict=checkin_dict,
        all_pois=all_pois,
        emb_path=emb_path,
        function_alpha=function_alpha
    )

    rmse_, mae_, mape_, smape_ = [], [], [], []

    for run_idx in range(args.runs):
        print(f"\n========== Run {run_idx + 1}/{args.runs} ==========\n")

        np.random.shuffle(all_pois)
        n = len(all_pois)

        train_ids = all_pois[:int(0.6 * n)]
        val_ids = all_pois[int(0.6 * n):int(0.8 * n)]
        test_ids = all_pois[int(0.8 * n):]

        train_dataset = PreprocessedPOIDataset(train_ids, checkin_dict, subgraph_dict, mode='train')
        val_dataset = PreprocessedPOIDataset(val_ids, checkin_dict, subgraph_dict, mode='val')
        test_dataset = PreprocessedPOIDataset(test_ids, checkin_dict, subgraph_dict, mode='test')

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.bs,
            shuffle=True,
            collate_fn=custom_collate,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.bs,
            shuffle=True,
            collate_fn=custom_collate,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.bs,
            shuffle=True,
            collate_fn=custom_collate,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        gnn_encoder = POIGAT(in_channels=768, hidden_channels=64, out_channels=768)
        dragonnet = DragonNetPredictor(input_dim=256, hidden_dim=128, out_weeks=4)
        forecast_model = GATGRUDragonModel(
            gnn=gnn_encoder,
            dragonnet=dragonnet,
            poi_dict=poi_dict
        )

        this_rmse, this_mae, this_mape, this_smape = train_regression_model(
            forecast_model,
            train_loader,
            val_loader,
            test_loader
        )

        rmse_.append(this_rmse)
        mae_.append(this_mae)
        mape_.append(this_mape)
        smape_.append(this_smape)

    print("\nFinal results out of " + str(args.runs) + ' runs:\n')
    print(
        'RMSE: ' + str(round(float(np.mean(np.array(rmse_))), 2)) +
        ' (' + str(round(float(np.std(np.array(rmse_))), 2)) + ') |' +
        ' MAE: ' + str(round(float(np.mean(np.array(mae_))), 2)) +
        ' (' + str(round(float(np.std(np.array(mae_))), 2)) + ') | ' +
        'MAPE: ' + str(round(float(np.mean(np.array(mape_))), 2)) +
        '% (' + str(round(float(np.std(np.array(mape_))), 2)) + '%) | ' +
        'SMAPE: ' + str(round(float(np.mean(np.array(smape_))), 2)) +
        '% (' + str(round(float(np.std(np.array(smape_))), 2)) + '%)'
    )


if __name__ == '__main__':
    main()

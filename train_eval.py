import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import config
from utils import print_epoch, mape, smape


def conditional_causal_loss(y0, y1, t_logit, y_true, treatment):
    t_pred = torch.sigmoid(t_logit).squeeze()
    treat_loss = nn.BCELoss()(t_pred, treatment.float())

    y_pred = treatment.unsqueeze(1) * y1 + (1 - treatment.unsqueeze(1)) * y0
    reg_loss = nn.MSELoss()(y_pred, y_true)

    return reg_loss + config.lamada * treat_loss, y_pred


def validate_regression(model, loader, testing=False, epoch=None):
    print(config.sep_width * "-")

    if not testing:
        print('Validation - Epoch: ' + str(epoch + 1))
    else:
        print('Testing best model...')

    print('(Regression Only)')
    print(config.sep_width * "-")

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for g_list, idx_list, treatment, y, poi_ids in loader:
            treatment = treatment.to(config.device)
            y = y.to(config.device)

            y0, y1, t_logit = model(g_list, idx_list, poi_ids)
            y_pred = treatment.unsqueeze(1) * y1 + (1 - treatment.unsqueeze(1)) * y0

            all_preds.append(y_pred.cpu())
            all_targets.append(y.cpu())

    y_true = torch.cat(all_targets, dim=0).cpu().numpy()
    y_pred = torch.cat(all_preds, dim=0).cpu().numpy()

    y_true = np.expm1(y_true)
    y_pred = np.expm1(y_pred)

    print("y_true mean:", np.mean(y_true), "min:", np.min(y_true), "max:", np.max(y_true))
    print("y_pred mean:", np.mean(y_pred), "min:", np.min(y_pred), "max:", np.max(y_pred))

    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape_value = mape(y_true, y_pred)
    smape_value = smape(y_true, y_pred)

    print(
        f'RMSE: {round(rmse, 2)} | MAE: {round(mae, 2)} | '
        f'MAPE: {round(mape_value, 2)}% | SMAPE: {round(smape_value, 2)}%'
    )
    print(config.sep_width * "-")

    if not testing:
        return rmse

    return rmse, mae, mape_value, smape_value


def train_regression_model(model, train_loader, valid_loader, test_loader,
                           training=True, device=config.device, epochs=config.epochs, lr=config.lr):
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    best_rmse = math.inf
    best_model_state = None

    for epoch in range(epochs):
        if not training:
            break

        print_epoch(epoch + 1, epochs)
        model.train()
        total_loss = 0.0

        for step, (g_list, idx_list, treatment, y, poi_ids) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        ):
            y = y.to(device)
            treatment = treatment.to(device)

            y0, y1, t_logit = model(g_list, idx_list, poi_ids)
            loss, _ = conditional_causal_loss(y0, y1, t_logit, y, treatment)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1} training loss: {round(total_loss / (step + 1), 4)}")

        rmse_ = validate_regression(model, valid_loader, testing=False, epoch=epoch)

        if rmse_ < best_rmse:
            best_rmse = rmse_
            best_model_state = model.state_dict()

        print(config.sep_width * "-")

    print("Final Evaluation on Test Set:")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return validate_regression(model, test_loader, testing=True, epoch=epochs)

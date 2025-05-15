"""
plot_surface.py

Plot embeddings and loss surface of model, assuming that the model
contains an embedding layer

Written by Will Solow, 2025
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from sklearn.decomposition import PCA

import utils

from model_engine.util import GRAPE_NAMES

def plot_loss_surface(calibrator, fpath, seed=0, high=1, low=-1):
    """
    Plots 2D and 3D renderings of loss surface
    """
    model_params = torch.cat([p.flatten() for p in calibrator.rnn.parameters()])
    torch.manual_seed(seed)

    # Create random directions
    dir1 = torch.randn_like(model_params)
    dir2 = torch.randn_like(model_params)
    dir1 /= torch.norm(dir1)
    dir2 /= torch.norm(dir2)

    # Define grid
    alpha = np.linspace(low, high, (high-low)*25)
    beta = np.linspace(low, high, (high-low)*25)
    loss_surface = np.zeros((len(alpha), len(beta)))

    with torch.no_grad():
        for k, a in enumerate(alpha):
            for j, b in enumerate(beta):
                # Perturb parameters
                new_params = model_params + a * dir1 + b * dir2
                idx = 0
                for p in calibrator.rnn.parameters():
                    numel = p.numel()
                    p.copy_(new_params[idx:idx+numel].view_as(p))
                    idx += numel

                # Compute loss
                for i in range(0, len(calibrator.data['train']), calibrator.batch_size):
                    calibrator.optimizer.zero_grad()
                    # Forward pass
                    output, _, _, _ = calibrator.forward(calibrator.data['train'][i:i+calibrator.batch_size], \
                                            calibrator.dates['train'][i:i+calibrator.batch_size], \
                                                calibrator.cultivars['train'][i:i+calibrator.batch_size] if calibrator.cultivars is not None else None)
                    # Get target
                    target = calibrator.val['train'][i:i+calibrator.batch_size]

                    # Compute and mask loss for backward pass
                    loss = calibrator.loss_func(output, target.nan_to_num(nan=0.0))
                    mask = ~torch.isnan(target)
                    loss = (loss * mask).sum() / mask.sum()
                    loss_surface[k, j] += loss.item()
                    
    # Plot 2D and 3D renderings of loss surface
    A, B = np.meshgrid(alpha, beta)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    np.savez(f"{fpath}/Loss_Surface_data_{seed}.npz", A=A, B=B, SURFACE=loss_surface)
    ax.plot_surface(A, B, loss_surface, cmap='viridis')
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss')
    plt.title('Loss Surface')
    plt.savefig(f"{fpath}/Loss_Surface_3D_{seed}.png", bbox_inches='tight')
    plt.close()

    plt.figure()
    cp = plt.contourf(A, B, loss_surface, levels=50, cmap='viridis')
    plt.colorbar(cp)
    plt.title('Loss Surface Contour Plot')
    plt.xlabel('Direction 1')
    plt.ylabel('Direction 2')
    plt.grid(True)
    plt.savefig(f"{fpath}/Loss_Surface_2D_{seed}.png", bbox_inches='tight')

def plot_cultivar_embeddings(calibrator, fpath):
    """
    Plots 2D and 3D embeddings of cultivars
    """
    
    embeddings = calibrator.rnn.embedding_layer(torch.arange(calibrator.num_cultivars).to(calibrator.device).to(torch.long)).detach().cpu().numpy()
    pca = PCA(n_components=2)
    pca_embeds = pca.fit_transform(embeddings)

    plt.figure()
    plt.title("2 Dimensional PCA Cultivar Embedding")
    plt.scatter(pca_embeds[:,0], pca_embeds[:,1])
    for i in range(len(GRAPE_NAMES['grape_phenology'])):
        plt.text(pca_embeds[i,0] + 0.02, pca_embeds[i,1] + 0.02, GRAPE_NAMES['grape_phenology'][i], fontsize=10)
    plt.savefig(f"{fpath}/Cultivar_Embeddings_2D.png",bbox_inches='tight')
    plt.close()

    pca = PCA(n_components=3)
    pca_embeds = pca.fit_transform(embeddings)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3 Dimensional PCA Cultivar Embedding")
    ax.scatter(pca_embeds[:,0], pca_embeds[:,1], pca_embeds[:,2] )
    for i in range(len(GRAPE_NAMES['grape_phenology'])):
        ax.text(pca_embeds[i,0] + 0.02, pca_embeds[i,1] + 0.02, pca_embeds[i,2], GRAPE_NAMES['grape_phenology'][i], fontsize=10)
    plt.savefig(f"{fpath}/Cultivar_Embeddings_3D.png",bbox_inches='tight')
    plt.close()

def main():

    argparser = argparse.ArgumentParser(description="Plotting script for LSTM model")
    argparser.add_argument("--config", type=str, required=True, help="Path to the config file")
    argparser.add_argument("--break_early", type=bool, default=False, help="If to break early when making data")
    argparser.add_argument("--save", type=bool, default=False, help="If to save the plots")
    argparser.add_argument("--split", default=3, type=int, help="data split")
    argparser.add_argument("--surface_seed", default=0, type=int)
    argparser.add_argument("--high", default=1,type=int)
    argparser.add_argument("--low", default=-1, type=int)


    args = argparser.parse_args()
    fpath = f"{os.getcwd()}/{args.config}"

    config, data, fpath = utils.load_config_data_fpath(args)

    calibrator = utils.load_rnn_from_config(config, args, data)
    
    assert "Embed" in config.RNN.rnn_arch, "Model architecture must contain an embedding layer"

    calibrator.load_model(f"{fpath}/rnn_model.pt")

    plot_cultivar_embeddings(calibrator,fpath)
    plot_loss_surface(calibrator,fpath, seed=args.surface_seed, high=args.high, low=args.low)

if __name__ == "__main__":
    main()
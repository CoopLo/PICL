import sys
import os
import torch
import numpy as np
import pickle
import shutil
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

# torch.manual_seed(0)
# np.random.seed(0)


sys.path.append('.')
#from models.oformer import Encoder1D, STDecoder1D, OFormer1D
from models.fno import FNO1d
#from models.deeponet import DeepONet1D
from utils import TransformerOperatorDataset, TransformerMultiOperatorDataset
from utils import WeightedNTXentLoss, PhysicsInformedWNTXentLoss, PhysicsInformedGCL, GCL, VICReg
from utils import PassthroughGCL

import yaml
from tqdm import tqdm
import h5py
from matplotlib import pyplot as plt

torch.autograd.detect_anomaly()


def progress_plots(ep, y_train_true, y_train_pred, y_val_true, y_val_pred, path="progress_plots", seed=None, dset=None):
    ncols = 4
    fig, ax = plt.subplots(ncols=ncols, nrows=2, figsize=(5*ncols,14))
    for i in range(ncols):
        ax[0][i].plot(y_train_true[i].reshape(50,).detach().cpu())
        ax[0][i].plot(y_train_pred[i].reshape(50,).detach().cpu())
        ax[1][i].plot(y_val_true[i].reshape(50,).detach().cpu())
        ax[1][i].plot(y_val_pred[i].reshape(50,).detach().cpu())

    fname = str(ep)
    while(len(fname) < 8):
        fname = '0' + fname
    if(seed is not None):
        if(dset is not None):
            plt.savefig("./{}/{}_{}_{}.png".format(path, dset, seed, fname))
        else:
            plt.savefig("./{}/{}_{}.png".format(path, seed, fname))
    else:
        plt.savefig("./{}/{}.png".format(path, fname))
    plt.close()


def val_plots(ep, val_loader, preds, path="progress_plots", seed=None):
    im_num = 0
    for vals in val_loader:
        for idx, v in tqdm(enumerate(vals[1])):

            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(v.reshape(200,).detach().cpu())
            #print(preds[0].shape)
            #ax.plot(preds[0][idx,:,0,0].detach().cpu())
            ax.plot(preds[0][idx].detach().cpu())
            fname = str(im_num)
            while(len(fname) < 8):
                fname = '0' + fname
            ax.set_title(fname)
            plt.savefig("./val_1/{}_{}.png".format(seed, fname))
            plt.close()

            im_num += 1


def train_plots(train_loader, seed=None):
    im_num = 0
    for vals in train_loader:
        #print(vals[2][0][1:] - vals[2][0][:-1])
        #raise
        for idx, v in tqdm(enumerate(vals[0])):

            fig, ax = plt.subplots(figsize=(8,6))
            for j in range(0, v.shape[0], 10):
                ax.plot(v[j].detach().cpu())
            fname = str(im_num)
            while(len(fname) < 8):
                fname = '0' + fname
            ax.set_title(fname)
            plt.savefig("./train_plots/{}_{}.png".format(seed, fname))
            plt.close()
            #raise

            im_num += 1


def get_model(model_name, config, finetune=False):
    if(model_name == "fno"):
        if(finetune):
            model = FNO1d(config['num_channels'], config['modes'], config['width'], config['initial_step']+4)#, dropout=config['finetune_dropout'])
        else: # No Dropout
            model = FNO1d(config['num_channels'], config['modes'], config['width'], config['initial_step']+4)#, dropout=config['dropout']) 
    elif(model_name == "oformer"):
        encoder = Encoder1D(input_channels=config['input_channels'], in_emb_dim=config['in_emb_dim'],
                            out_seq_emb_dim=config['out_seq_emb_dim'], depth=config['depth'], dropout=config['dropout'],
                            res=config['enc_res'])
        decoder = STDecoder1D(latent_channels=config['latent_channels'], out_channels=config['out_channels'],
                                     decoding_depth=config['decoding_depth'], scale=config['scale'], res=config['dec_res'])
        model = OFormer1D(encoder, decoder)
    elif(model_name == "deeponet"):
        model = DeepONet1D(config['branch_net'], config['trunk_net'], config['activation'], config['kernel_initializer'])
    
    model.to(device)
    return model


def get_data(f, config, pretraining=False):
    if(config['flnm'] == 'all'):
        #train_data = TransformerOperatorDataset(f, config['flnm'],
        train_data = TransformerMultiOperatorDataset(config['base_path'],
                                split="train",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                return_text=config['return_text'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['pretraining_num_samples'] if(pretraining) else config['num_samples'],
                                train_style=config['train_style'],
                                rollout_length=config['rollout_length'],
                                seed=config['seed'],
                                forcing_term=config['forcing_term'],
                                flnm=config['flnm'],
        )
        train_data.data = train_data.data.to(device)
        train_data.grid = train_data.grid.to(device)
        #train_data.time_included_tokens = train_data.time_included_tokens.to(device)
        #val_data = TransformerOperatorDataset(f, config['flnm'],
        val_data = TransformerMultiOperatorDataset(config['base_path'],
                                split="val",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                return_text=config['return_text'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                #num_samples=config['num_samples'],
                                num_samples=config['pretraining_num_samples'] if(pretraining) else config['num_samples'],
                                train_style=config['train_style'],
                                rollout_length=config['rollout_length'],
                                seed=config['seed'],
                                forcing_term=config['forcing_term'],
                                flnm=config['flnm'],
        )
        val_data.data = val_data.data.to(device)
        val_data.grid = val_data.grid.to(device)
        #val_data.time_included_tokens = val_data.time_included_tokens.to(device)
        #test_data = TransformerOperatorDataset(f, config['flnm'],
        test_data = TransformerMultiOperatorDataset(config['base_path'],
                                split="test",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                return_text=config['return_text'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                #num_samples=config['num_samples'],
                                num_samples=config['pretraining_num_samples'] if(pretraining) else config['num_samples'],
                                train_style=config['train_style'],
                                rollout_length=config['rollout_length'],
                                seed=config['seed'],
                                forcing_term=config['forcing_term'],
                                flnm=config['flnm'],
        )
        test_data.data = test_data.data.to(device)
        test_data.grid = test_data.grid.to(device)
        #test_data.time_included_tokens = test_data.time_included_tokens.to(device)
    else:
        train_data = TransformerOperatorDataset(f, config['flnm'],
                                split="train",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                return_text=config['return_text'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['num_samples'],
                                train_style=config['train_style'],
                                rollout_length=config['rollout_length'],
                                seed=config['seed'],
                                #forcing_term=config['forcing_term'],
        )
        train_data.data = train_data.data.to(device)
        train_data.grid = train_data.grid.to(device)
        train_data.all_tokens = train_data.all_tokens.to(device)
        train_data.all_operator_maps = train_data.all_operator_maps.to(device)
        train_data.time = train_data.time.to(device)
        #train_data.time_included_tokens = train_data.time_included_tokens.to(device)
        val_data = TransformerOperatorDataset(f, config['flnm'],
                                split="val",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                return_text=config['return_text'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['num_samples'],
                                train_style=config['train_style'],
                                rollout_length=config['rollout_length'],
                                seed=config['seed'],
                                #forcing_term=config['forcing_term'],
        )
        val_data.data = val_data.data.to(device)
        val_data.grid = val_data.grid.to(device)
        test_data = TransformerOperatorDataset(f, config['flnm'],
                                split="test",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                return_text=config['return_text'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['num_samples'],
                                train_style=config['train_style'],
                                rollout_length=config['rollout_length'],
                                seed=config['seed'],
                                #forcing_term=config['forcing_term'],
        )
        test_data.data = test_data.data.to(device)
        test_data.grid = test_data.grid.to(device)
        #test_data.time_included_tokens = test_data.time_included_tokens.to(device)

    if(pretraining):
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=config['pretraining_batch_size'],
                                                   num_workers=config['num_workers'], shuffle=True)#,
                                                   #generator=torch.Generator(device='cuda'))
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=config['pretraining_batch_size'],
                                                 num_workers=config['num_workers'], shuffle=False)#,
                                                 #generator=torch.Generator(device='cuda'))
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['pretraining_batch_size'],
                                                 num_workers=config['num_workers'], shuffle=False)#,
                                             #generator=torch.Generator(device='cuda'))
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=config['batch_size'],
                                                   num_workers=config['num_workers'], shuffle=True)#,
                                                   #generator=torch.Generator(device='cuda'))
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=config['batch_size'],
                                                 num_workers=config['num_workers'], shuffle=False)#,
                                                 #generator=torch.Generator(device='cuda'))
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'],
                                                 num_workers=config['num_workers'], shuffle=False)#,

    assert not (bool(set(train_data.data_list) & \
                     set(val_data.data_list)) | \
                bool(set(train_data.data_list) & \
                     set(test_data.data_list)) & \
                bool(set(val_data.data_list) & \
                     set(test_data.data_list)))

    return train_loader, val_loader, test_loader


def get_single_data(f, config):
    if(config['flnm'] == 'all'):
        #train_data = TransformerOperatorDataset(f, config['flnm'],
        train_data = TransformerMultiOperatorDataset(config['base_path'],
                                split="train",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                return_text=config['return_text'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['num_samples'],
                                train_style=config['train_style'],
                                rollout_length=config['rollout_length'],
                                seed=config['seed'],
                                forcing_term=config['forcing_term'],
                                flnm=config['flnm'],
        )
        train_data.data = train_data.data.to(device)
        train_data.grid = train_data.grid.to(device)
        #train_data.time_included_tokens = train_data.time_included_tokens.to(device)
        #val_data = TransformerOperatorDataset(f, config['flnm'],
        val_data = TransformerMultiOperatorDataset(config['base_path'],
                                split="val",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                return_text=config['return_text'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['num_samples'],
                                train_style=config['train_style'],
                                rollout_length=config['rollout_length'],
                                seed=config['seed'],
                                forcing_term=config['forcing_term'],
                                flnm=config['flnm'],
        )
        val_data.data = val_data.data.to(device)
        val_data.grid = val_data.grid.to(device)
        #val_data.time_included_tokens = val_data.time_included_tokens.to(device)
        #test_data = TransformerOperatorDataset(f, config['flnm'],
        test_data = TransformerMultiOperatorDataset(config['base_path'],
                                split="test",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                return_text=config['return_text'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['num_samples'],
                                train_style=config['train_style'],
                                rollout_length=config['rollout_length'],
                                seed=config['seed'],
                                forcing_term=config['forcing_term'],
                                flnm=config['flnm'],
        )
        test_data.data = test_data.data.to(device)
        test_data.grid = test_data.grid.to(device)
        #test_data.time_included_tokens = test_data.time_included_tokens.to(device)
    else:
        train_data = TransformerOperatorDataset(f, config['flnm'],
                                split="train",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                return_text=config['return_text'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['num_samples'],
                                train_style=config['train_style'],
                                rollout_length=config['rollout_length'],
                                seed=config['seed'],
                                #forcing_term=config['forcing_term'],
        )
        train_data.data = train_data.data.to(device)
        train_data.grid = train_data.grid.to(device)
        #train_data.time_included_tokens = train_data.time_included_tokens.to(device)
        val_data = TransformerOperatorDataset(f, config['flnm'],
                                split="val",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                return_text=config['return_text'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['num_samples'],
                                train_style=config['train_style'],
                                rollout_length=config['rollout_length'],
                                seed=config['seed'],
                                #forcing_term=config['forcing_term'],
        )
        val_data.data = val_data.data.to(device)
        val_data.grid = val_data.grid.to(device)
        test_data = TransformerOperatorDataset(f, config['flnm'],
                                split="test",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                return_text=config['return_text'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['num_samples'],
                                train_style=config['train_style'],
                                rollout_length=config['rollout_length'],
                                seed=config['seed'],
                                #forcing_term=config['forcing_term'],
        )
        test_data.data = test_data.data.to(device)
        test_data.grid = test_data.grid.to(device)
        #test_data.time_included_tokens = test_data.time_included_tokens.to(device)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config['batch_size'],
                                               num_workers=config['num_workers'], shuffle=True)#,
                                               #generator=torch.Generator(device='cuda'))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config['batch_size'],
                                             num_workers=config['num_workers'], shuffle=False)#,
                                             #generator=torch.Generator(device='cuda'))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'],
                                             num_workers=config['num_workers'], shuffle=False)#,
                                             #generator=torch.Generator(device='cuda'))

    assert not (bool(set(train_data.data_list) & \
                     set(val_data.data_list)) | \
                bool(set(train_data.data_list) & \
                     set(test_data.data_list)) & \
                bool(set(val_data.data_list) & \
                     set(test_data.data_list)))

    return train_loader, val_loader, test_loader


def get_rollout_eval_data(f, config):
    if(config['flnm'] == 'all'):
        test_data = TransformerMultiOperatorDataset(config['base_path'],
                                split="test",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                return_text=config['return_text'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=5000,
                                train_style=config['train_style'],
                                rollout_length=config['rollout_length'],
                                seed=config['seed'],
                                forcing_term=config['forcing_term'],
                                flnm=config['flnm'],
        )
        test_data.data = test_data.data.to(device)
        test_data.grid = test_data.grid.to(device)
    else:
        test_data = TransformerOperatorDataset(f, config['flnm'],
                                split="test",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                return_text=config['return_text'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=5000,
                                train_style=config['train_style'],
                                rollout_length=config['rollout_length'],
                                seed=config['seed'],
                                #forcing_term=config['forcing_term'],
        )
        test_data.data = test_data.data.to(device)
        test_data.grid = test_data.grid.to(device)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'],
                                             num_workers=config['num_workers'], shuffle=False)#,
                                             #generator=torch.Generator(device='cuda'))

    assert not (bool(set(train_data.data_list) & \
                     set(val_data.data_list)) | \
                bool(set(train_data.data_list) & \
                     set(test_data.data_list)) & \
                bool(set(val_data.data_list) & \
                     set(test_data.data_list)))

    return train_loader, val_loader, test_loader


def evaluate(test_loader, model, loss_fn):
    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        model.eval()
        #for bn, (xx, yy, grid) in enumerate(test_loader):
        for bn, (xx, y, grid, tokens, t, target) in enumerate(test_loader):
            
            #if(isinstance(model, (FNO1d, OFormer1D))):
            if(isinstance(model, FNO1d)):
                #x = torch.swapaxes(xx, 1, 2)
                #grid = torch.swapaxes(grid, 1, 2)
                #grid = grid.unsqueeze(-1)#torch.swapaxes(grid, 1, 2)
                ##im, loss = model.get_loss(x, yy[:,0,:], grid, loss_fn)
                #if(len(yy.shape) == 3):
                #    yy = yy[...,0]
                #im, loss = model.get_loss(x, yy, grid, loss_fn)
                x = torch.swapaxes(xx, 1, 2)
                grid = torch.swapaxes(grid.unsqueeze(1), 1, 2)
                t = t.cuda()
                target = target.cuda()

                # Standard forward pass
                x = torch.swapaxes(xx, 1, 2).cuda()
                im = model(x, grid, t=t, coeffs=target)[...,0]
                loss = loss_fn(im, y)
            elif(isinstance(model,DeepONet1D)):
                x = torch.swapaxes(xx, 1, 2)
                grid = torch.swapaxes(grid.unsqueeze(-1), 1, 2)
                im = model(x, grid)[...,0]
                if(len(yy.shape) == 3 and len(im.shape) == 2):
                    im = im.unsqueeze(-1)
                loss = loss_fn(yy, im)

            test_l2_step += loss.item()
            test_l2_full += loss.item()
    return test_l2_full/(bn+1)


def as_rollout_eval(config, path, model_path, prefix, seed=None, test_loader=None):
    model.load_state_dict(torch.load(model_path)['model_state_dict'])

    model.eval()
    all_y_preds = []
    all_y_trues = []
    print("Rollout eval...")
    with torch.no_grad():
        for i in tqdm(range(test_loader.dataset.data.shape[0])):
            # Get data
            x0 = test_loader.dataset.data[i][:config['initial_step']]
            y = test_loader.dataset.data[i][config['initial_step']:]
            t = test_loader.dataset.time[i][config['initial_step']:]
            grid = test_loader.dataset.grid[i]
            coeffs = test_loader.dataset.all_operator_maps[i]

            # Make it the right shape
            x = x0.unsqueeze(0).repeat(y.shape[0],1,1).transpose(1,2)
            grid = grid.unsqueeze(0).broadcast_to(y.shape[0], grid.shape[0]).to(device)
            coeffs = coeffs.unsqueeze(0).broadcast_to(y.shape[0], coeffs.shape[0]).to(device)

            t = t.to(device)
            y = y.unsqueeze(-1).to(device)

            y_pred = model(x, grid.unsqueeze(-1), t=t, coeffs=coeffs)[...,0,0]

            all_y_preds.append(y_pred.unsqueeze(0))
            all_y_trues.append(y.unsqueeze(0)[...,0])

    all_y_preds = torch.cat(all_y_preds, dim=0)
    all_y_trues = torch.cat(all_y_trues, dim=0)

    # Now in shape traj x time x space x channels
    mse = ((all_y_preds - all_y_trues)**2).mean(dim=(0,2))

    # Save relevant info
    dname = 'all' if(isinstance(test_loader.dataset, TransformerMultiOperatorDataset)) else \
            'heat' if('heat' in test_loader.dataset.file_path) else \
            'burgers' if('burgers' in test_loader.dataset.file_path) else \
            'adv' if ('adv' in test_loader.dataset.file_path) else None
    if(dname is None):
        raise ValueError("Issue with dataset file path.")

    print("\n\nPATH: {}\n\n".format(path))
    torch.save(mse, path+"/{}_rollout_mse".format(seed))
    torch.save(all_y_trues.cpu(), path+"/{}_{}_y_trues".format(seed, dname))
    torch.save(all_y_preds.cpu(), path+"/{}_{}_y_preds".format(seed, dname))


def ar_rollout_eval(config, path, model_path, prefix, seed=None, test_loader=None):
    model.load_state_dict(torch.load(model_path)['model_state_dict'])

    model.eval()
    all_y_preds = []
    all_y_trues = []
    with torch.no_grad():
        for i in tqdm(range(test_loader.dataset.data.shape[0])):
            # Get data
            x0 = test_loader.dataset.data[i][:config['initial_step']]
            y = test_loader.dataset.data[i][config['initial_step']:]
            t = test_loader.dataset.time[i][config['initial_step']:]
            grid = test_loader.dataset.grid[i]
            coeffs = test_loader.dataset.all_operator_maps[i]

            # Make it the right shape
            x = x0.unsqueeze(0).transpose(1,2)
            grid = grid.unsqueeze(0).to(device)
            coeffs = coeffs.unsqueeze(0).to(device)

            y_preds = []
            y_trues = []
            for i in range(y.shape[0]):
                inp_t = t[i].to(device).unsqueeze(0)
                y_pred = model(x, grid.unsqueeze(-1), t=inp_t, coeffs=coeffs)[...,0,0]
                y_preds.append(y_pred.unsqueeze(0))
                y_trues.append(y[i].unsqueeze(0).unsqueeze(0))
                x = torch.cat((x, y_pred.unsqueeze(-1)), dim=-1)[...,-config['initial_step']:]

            all_y_preds.append(torch.cat(y_preds, dim=1))
            all_y_trues.append(torch.cat(y_trues, dim=1))
            
    
    all_y_preds = torch.cat(all_y_preds, dim=0)
    all_y_trues = torch.cat(all_y_trues, dim=0)

    # Now in shape traj x time x space x channels
    mse = ((all_y_preds - all_y_trues)**2).mean(dim=(0,2))
    
    # Save relevant info
    dname = 'all' if(isinstance(test_loader.dataset, TransformerMultiOperatorDataset)) else \
            'heat' if('heat' in test_loader.dataset.file_path) else \
            'burgers' if('burgers' in test_loader.dataset.file_path) else \
            'adv' if ('adv' in test_loader.dataset.file_path) else None
    if(dname is None):
        raise ValueError("Issue with dataset file path.")

    torch.save(mse, path+"/{}_{}_rollout_mse".format(seed, dname))
    torch.save(all_y_trues.cpu(), path+"/{}_{}_y_trues".format(seed, dname))
    torch.save(all_y_preds.cpu(), path+"/{}_{}_y_preds".format(seed, dname))
    torch.save(mse, path+"/{}_{}_rollout_mse".format(seed, dname))


def load_model(model, config, prefix, seed):
    
    ################################################################
    # load data
    ################################################################
    
    #print()
    #print()
    #print(train_args['results_dir'])
    #print()
    #print()
    #raise
    #results_dir =  "./CONTAINS_PRETRAINED_MODELS/100/"
    #results_dir =  "./PRETRAINED_MODELS/{}/".format(config['pretraining_num_samples'])
    results_dir =  "./PASSTHROUGH_FIXED_PRETRAINED_MODELS/{}/".format(config['pretraining_num_samples'])
    path = "{}pretrain_{}_{}_{}_{}".format(results_dir, config['model_name'], config['contrastive_loss'], config['similarity'], prefix)
    #f = h5py.File("{}{}".format(config['base_path'], config['data_name']), 'r')
    model_name = 'pretrain_' + config['flnm'] + '_{}'.format(config['model_name']) + "_{}.pt".format(seed)
    model_path = path + "/" + model_name
    
    #print("Filename: {}, Seed: {}\n".format(config['flnm'], config['seed']))

    #train_loader, val_loader, test_loader = get_data(f, config, pretraining=False)
    #train_loader.dataset.pretrain()
    #val_loader.dataset.pretrain()
    #test_loader.dataset.pretrain()

    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    print("SUCCESSFULLY LOADED PRETRAINED MODEL")

    return model, None, None, None, model_path


def pretraining(model, config, prefix):
    
    ################################################################
    # load data
    ################################################################
    
    path = "{}pretrain_{}_{}_{}_{}".format(train_args['results_dir'], config['model_name'], config['contrastive_loss'], config['similarity'], prefix)
    f = h5py.File("{}{}".format(config['base_path'], config['data_name']), 'r')
    model_name = 'pretrain_' + config['flnm'] + '_{}'.format(config['model_name']) + "_{}.pt".format(seed)
    model_path = path + "/" + model_name
    
    print("Filename: {}, Seed: {}\n".format(config['flnm'], config['seed']))

    train_loader, val_loader, test_loader = get_data(f, config, pretraining=True)
    train_loader.dataset.pretrain()
    val_loader.dataset.pretrain()
    test_loader.dataset.pretrain()
    #if(config['load']):
    #    return model, train_loader, val_loader, test_loader, model_path
    #raise

    if(config['pretrain_done']):
        return model, train_loader, val_loader, test_loader, model_path
    
    ################################################################
    # training and evaluation
    ################################################################
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')
    # Use Adam for Hyena
    optimizer = torch.optim.Adam(model.parameters(), lr=config['pretrain_learning_rate'],
                                 weight_decay=config['pretrain_weight_decay'])
    print("WEIGHT DECAY: {}".format(config['pretrain_weight_decay']))
    print("\nUSING STEP SCHEDULER\n")
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['pretrain_scheduler_step'],
    #                                            gamma=config['pretrain_scheduler_gamma'])
    #if(False and isinstance(model, OFormer1D)):
    if(False and isinstance(model)):
        print("\nUSING ONECYCLELER SCHEDULER\n")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['pretrain_learning_rate'],# div_factor=1e6,
                                                        steps_per_epoch=len(train_loader), epochs=config['epochs'])
    else:
        print("\nUSING STEPLR SCHEDULER\n")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, max_lr=config['pretrain_learning_rate'],
                            steps_per_epoch=len(train_loader), epochs=config['pretrain_epochs']
        )
        #scheduler = torch.optim.lr_scheduler.StepLR(
        #                    optimizer, step_size=config['pretrain_scheduler_step'],
        #                    gamma=config['pretrain_scheduler_gamma']
        #)

    #classification_layer = nn.Linear(300, 2, bias=False)
    #cl_optimizer = torch.optim.Adam(classification_layer.parameters(), lr=0.01*config['learning_rate'], weight_decay=config['weight_decay'])
    #cl_optimizer = torch.optim.Adam(classification_layer.parameters(), lr=config['learning_rate'], weight_decay=0)
    #cl_scheduler = torch.optim.lr_scheduler.OneCycleLR(cl_optimizer, max_lr=config['learning_rate'],# div_factor=1e6,
    #                                                steps_per_epoch=len(train_loader), epochs=10)
    #cl_scheduler = torch.optim.lr_scheduler.StepLR(cl_optimizer, step_size=config['scheduler_step'], gamma=config['scheduler_gamma'])

    
    #loss_fn = nn.L1Loss(reduction="mean")
    #loss_fn = nn.CrossEntropyLoss(reduction="mean")
    if(config['contrastive_loss'] == 'GCL'):
        loss_fn = PhysicsInformedGCL('cuda')
        print("\nCONTRASTIVE LOSS: {}".format(loss_fn))
    elif(config['contrastive_loss'] == 'normalGCL'):
        loss_fn = GCL('cuda', tau=config['tau'])
        print("\nCONTRASTIVE LOSS: {}".format(loss_fn))
    elif(config['contrastive_loss'] == 'passthroughGCL'):
        loss_fn = PassthroughGCL('cuda', tau=config['tau'])
    elif(config['contrastive_loss'] == 'wnxent'):
        loss_fn = WeightedNTXentLoss('cuda', similarity=config['similarity'])
        print("\nCONTRASTIVE LOSS: {}".format(loss_fn))
    elif(config['contrastive_loss'] == 'physics_informed'):
        loss_fn = PhysicsInformedWNTXentLoss('cuda')
        print("\nCONTRASTIVE LOSS: {}".format(loss_fn))
    elif(config['contrastive_loss'] == 'vicreg'):
        loss_fn = VICReg()
    else:
        raise ValueError("\nINVALID CONSTRASTIVE LOSS FUNCTION CHOICE\n")
    loss_val_min = np.infty
    
    start_epoch = 0
    
    train_l2s, val_l2s = [], []
    print("\nPRETRAINING...")
    dt = (train_loader.dataset.time[0][1] - train_loader.dataset.time[0][0]).unsqueeze(0)
    for ep in tqdm(range(config['pretrain_epochs'])):
        #model.pretrain()
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        #for bn, (xx, yy, grid) in enumerate(train_loader):
        train_acc = 0
        num_samples = 0
        #for v in train_loader:
        #    print(len(v))
        #    raise
        for bn, (xx, ys, grid, tokens, t, target) in enumerate(train_loader):
            
            # Each model handles input differnetly
            if(isinstance(model, FNO1d)):
                #print()
                #print()
                #print(t.shape, tokens.shape, target.shape)
                #print()
                #print()
                #raise
                grid = torch.swapaxes(grid.unsqueeze(1), 1, 2)
                inp_dt = dt.broadcast_to(t.shape).cuda()
                t = t.cuda()
                target = target.cuda()

                # Standard forward pass
                x = torch.swapaxes(xx, 1, 2).cuda()
                y_pred = model(x, grid, t=t, coeffs=target)

                # Spatiall shift points
                ##shift = torch.randint(low=-10, high=10, size=(1,)).item()
                ##x_aug = torch.roll(x, shift, 1)
                ##y_pred_aug = model(x_aug, grid)

                ### Stack
                ##y_pred_aug = torch.roll(y_pred_aug, -shift, 1)
                ##y_pred = torch.cat((y_pred, y_pred_aug), dim=0)
                ##x = torch.cat((x,x), dim=0)
                ##grid = torch.cat((grid, grid), dim=0)
                ##t = torch.cat((t,t))
                ##target = torch.cat((target, target), dim=0)

                if(config['contrastive_loss'] in ['physics_informed', 'GCL', 'normalGCL', 'passthroughGCL']):
                    dx = grid[0][1] - grid[0][0]
                    #loss = loss_fn(y_pred.cuda(), y_pred.cuda(), target.float(), x.cuda(), t.cuda(), dx.cuda())
                    loss = loss_fn(y_pred.cuda(), y_pred.cuda(), target.cuda(), x.cuda(), inp_dt.cuda(), dx.cuda())
                else:
                    dx = grid[0][1] - grid[0][0]
                    loss = loss_fn(y_pred, y_pred, x, dt, dx, target)
                    #loss = loss_fn(y_pred, y_pred, target)

                # Backward pass: compute gradient of the loss with respect to model
                # parameters.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            elif(isinstance(model,DeepONet1D)):
                x = torch.swapaxes(xx, 1, 2)
                grid = torch.swapaxes(grid.unsqueeze(-1), 1, 2)
                im = model(x, grid)[...,0,0]
                y_pred = model(x, grid).unsqueeze(-1)
                if(config['contrastive_loss'] in ['physics_informed', 'GCL', 'passthroughGCL']):
                    dx = grid[0,:,1] - grid[0,:,0]
                    grid = torch.swapaxes(grid, 1, 2)
                    loss = loss_fn(y_pred, y_pred, target, x, t, dx)
                else:
                    dx = grid[0,:,1] - grid[0,:,0]
                    loss = loss_fn(y_pred, y_pred, target, x, t, dx)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_l2_step += loss.item()
            train_l2_full += loss.item()

            if(bn%100 == 0):
                print("Batch {0}/{1}: {2:.5f}".format(bn, len(train_loader), train_l2_full/(bn+1)))
        

        #print(pred)
        train_l2s.append(train_l2_full/(bn+1))
        if ep % config['validate'] == 0:
            val_l2_step = 0
            val_l2_full = 0
            with torch.no_grad():
                model.eval()
                #for bn, (xx, yy, grid) in enumerate(val_loader):
                val_acc = 0
                val_samples = 0
                for bn, (xx, ys, grid, tokens, t, target) in enumerate(val_loader):

                    # Each model handles input differnetly
                    if(isinstance(model, FNO1d)):
                        x = torch.swapaxes(xx, 1, 2)
                        grid = torch.swapaxes(grid.unsqueeze(1), 1, 2)
                        inp_dt = dt.broadcast_to(t.shape).cuda()
                        t = t.cuda()
                        target = target.cuda()

                        #y_pred = model(x, grid)
                        y_pred = model(x, grid, t=t, coeffs=target)
                        if(config['contrastive_loss'] in ['physics_informed', 'GCL', 'normalGCL', 'passthroughGCL']):
                            #loss = loss_fn(y_pred, y_pred, target.float(), x, t, dx)
                            loss = loss_fn(y_pred.cuda(), y_pred.cuda(), target.cuda(), x.cuda(), inp_dt.cuda(), dx.cuda())
                        else:
                            dx = grid[0][1] - grid[0][0]
                            loss = loss_fn(y_pred, y_pred, x, t/199, dx, target)
                            #loss = loss_fn(y_pred, y_pred, x, target)
                            #loss = loss_fn(y_pred, y_pred, target)

                    elif(isinstance(model,DeepONet1D)):
                        x = torch.swapaxes(xx, 1, 2)
                        grid = torch.swapaxes(grid.unsqueeze(-1), 1, 2)
                        im = model(x, grid)[...,0,0]
                        y_pred = model(x, grid)
                        y_pred = model(x, grid).unsqueeze(-1)
                        if(config['contrastive_loss'] in ['physics_informed', 'GCL', 'passthroughGCL']):
                            dx = grid[0,:,1] - grid[0,:,0]
                            grid = torch.swapaxes(grid, 1, 2)
                            loss = loss_fn(y_pred, y_pred, target, x, t, dx)
                        else:
                            dx = grid[0,:,1] - grid[0,:,0]
                            loss = loss_fn(y_pred, y_pred, target, x, t, dx)

                    # Guarantees we're able to plot at least a few from first batch
                    #if(bn == 0):
                    #    y_val_true = yy[:,0,:].clone()
                    #    y_val_pred = im.clone()

                    val_l2_step += loss.item()
                    val_l2_full += loss.item()
                
                #if  val_l2_full < loss_val_min:
                loss_val_min = val_l2_full
                best_ep = ep
                best_model = model.state_dict()
                best_optimizer = optimizer.state_dict()
                best_loss_val_min = loss_val_min

                # Save best
                torch.save({
                    'epoch': best_ep,
                    'model_state_dict': best_model,
                    'optimizer_state_dict': best_optimizer,
                    'loss': best_loss_val_min
                }, model_path)

        model.train()
        val_l2s.append(val_l2_full/(bn+1))
                
        t2 = default_timer()
        scheduler.step()
        if((ep%config['log_freq'] == 0) or (config['pretrain_epochs'] < 101)):
            print('epoch: {0}, loss: {1:.5f}, time: {2:.5f}s, trainL2: {3:.5f}, testL2: {4:.5f}'\
                .format(ep, loss.item(), t2 - t1, train_l2s[-1], val_l2s[-1]))
            #print("TRAIN ACCURACY: {0:.2f}% \t VALIDATION ACCURACY: {1:.2f}%".format(100*train_acc/num_samples,
            #                                                                         100*val_acc/val_samples))
            np.save("./{}/pretrain_train_l2s_{}.npy".format(path, seed), train_l2s)
            np.save("./{}/pretrain_val_l2s_{}.npy".format(path, seed), val_l2s)

        #if(ep%config['progress_plot_freq'] == 0 and len(y_train_true) >= 4):
        #    progress_plots(ep, y_train_true, y_train_pred, y_val_true, y_val_pred, path, seed=seed)


    # Make sure to capture last
    print('epoch: {0}, loss: {1:.5f}, time: {2:.5f}s, trainL2: {3:.5f}, testL2: {4:.5f}'\
          .format(ep, loss.item(), t2 - t1, train_l2s[-1], val_l2s[-1]))
    np.save("./{}/pretrain_train_l2s_{}.npy".format(path, seed), train_l2s)
    np.save("./{}/pretrain_val_l2s_{}.npy".format(path, seed), val_l2s)
    #progress_plots(ep, y_train_true, y_train_pred, y_val_true, y_val_pred, path, seed=seed)

    with torch.no_grad():
        model.eval()
        all_preds = torch.Tensor([]).cpu()
        all_targets = torch.Tensor([]).cpu()
        model._embed = True
        for bn, (x0, y, grid, tokens, t, target) in enumerate(train_loader):
            #x = torch.swapaxes(x0, 1, 2)
            #grid = torch.swapaxes(grid.unsqueeze(1), 1, 2)
            #y_pred = model(x, grid).detach().cpu()
            grid = torch.swapaxes(grid.unsqueeze(1), 1, 2)
            t = t.cuda()
            target = target.cuda()

            # Standard forward pass
            x = torch.swapaxes(x0, 1, 2).cuda()
            #print()
            #print(x.shape, grid.shape, t.shape, target.shape)
            #print()
            y_pred = model(x, grid, t=t, coeffs=target)
            all_preds = torch.cat((all_preds, y_pred.cpu()), dim=0)
            all_targets = torch.cat((all_targets, target.cpu()), dim=0)
    #print(all_preds.shape)
    if(seed == 0):
        np.save("./{}/pretrain_all_preds_{}.npy".format(path, seed), all_preds.cpu())
        np.save("./{}/pretrain_all_targets_{}.npy".format(path, seed), all_targets.cpu())
    model._embed = False

    #test_vals = []
    ##model.eval()
    #test_value = evaluate(test_loader, model, loss_fn)
    #test_vals.append(test_value)
    #print("TEST VALUE FROM LAST EPOCH: {0:5f}".format(test_value))
    #test_value = evaluate(test_loader, model, loss_fn)
    #test_vals.append(test_value)
    #print("TEST VALUE BEST LAST EPOCH: {0:5f}".format(test_value))
    #np.save("./{}/test_vals_{}.npy".format(path, seed), test_vals)
    #model.train()
    
    # Early stopping
    model.load_state_dict(torch.load(model_path)['model_state_dict'])

    # Save pretrained model
    #results_dir =  "./PRETRAINED_MODELS/{}/".format(config['pretraining_num_samples'])
    results_dir =  "./PASSTHROUGH_FIXED_PRETRAINED_MODELS/{}/".format(config['pretraining_num_samples'])
    path = "{}pretrain_{}_{}_{}_{}".format(results_dir, config['model_name'], config['contrastive_loss'], config['similarity'], prefix)
    model_name = 'pretrain_' + config['flnm'] + '_{}'.format(config['model_name']) + "_{}.pt".format(seed)
    model_path = path + "/" + model_name
    os.makedirs(path, exist_ok=True)

    torch.save({
        'epoch': best_ep,
        'model_state_dict': best_model,
        'optimizer_state_dict': best_optimizer,
        'loss': best_loss_val_min
    }, model_path)


    return model, train_loader, val_loader, test_loader, model_path


def run_training(old_model, config, prefix, train_loader, val_loader, test_loader):
    
    ################################################################
    # load data
    ################################################################
   
    path = "{}pretrain_{}_{}_{}_{}".format(train_args['results_dir'], config['model_name'], config['contrastive_loss'], config['similarity'], prefix)
    f = h5py.File("{}{}".format(config['base_path'], config['data_name']), 'r')
    model_name = config['flnm'] + '_{}'.format(config['model_name']) + "_{}.pt".format(seed)
    model_path = path + "/" + model_name
    
    print("Filename: {}, Seed: {}\n".format(config['flnm'], config['seed']))

    train_loader, val_loader, test_loader = get_data(f, config)
    # Reset batch size
    #train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=config['batch_size'],
    #                                           num_workers=config['num_workers'], shuffle=True)#,
    #                                           #generator=torch.Generator(device='cuda'))
    #val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=config['batch_size'],
    #                                         num_workers=config['num_workers'], shuffle=False)#,
    #                                         #generator=torch.Generator(device='cuda'))
    #test_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=config['batch_size'],
    #                                         num_workers=config['num_workers'], shuffle=False)#,
    #                                         #generator=torch.Generator(device='cuda'))
    #train_loader.dataset.pretrain_off()
    #train_loader.dataset.return_text = False
    #val_loader.dataset.pretrain_off()
    #val_loader.dataset.return_text = False
    #test_loader.dataset.pretrain_off()
    #test_loader.dataset.return_text = False

    model = get_model(config['model_name'], train_args)
    model.load_state_dict(old_model.state_dict())
    model.pretrain_off()
    
    ################################################################
    # training and evaluation
    ################################################################
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')
    # Use Adam for Hyena
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    #optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step'], gamma=config['scheduler_gamma'])
    
    loss_fn = nn.L1Loss(reduction="mean")
    #loss_fn = nn.MSELoss(reduction="mean")
    loss_val_min = np.infty
    
    start_epoch = 0
    
    train_l2s, val_l2s = [], []
    for ep in tqdm(range(start_epoch, config['epochs'])):
        model.train()
        #model.eval()
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        #for bn, (xx, yy, grid) in enumerate(train_loader):
        for bn, (xx, y, grid, tokens, t, target) in enumerate(train_loader):
            
            # Each model handles input differnetly
            #if(isinstance(model, (FNO1d, OFormer1D))):
            if(isinstance(model, FNO1d)):
                #print(xx.shape, grid.shape)
                #x = torch.swapaxes(xx, 1, 2)
                #grid = grid.unsqueeze(-1)#torch.swapaxes(grid, 1, 2)
                #if(len(yy.shape) == 3):
                #    yy = yy[...,0]
                #im, loss = model.get_loss(x, yy, grid, loss_fn)

                x = torch.swapaxes(xx, 1, 2)
                grid = torch.swapaxes(grid.unsqueeze(1), 1, 2)
                t = t.cuda()
                target = target.cuda()

                # Standard forward pass
                x = torch.swapaxes(xx, 1, 2).cuda()
                im = model(x, grid, t=t, coeffs=target)[...,0]
                loss = loss_fn(im, y)

                if(im.isnan().any()):
                    for param in model.parameters():
                        print("NANS IN WEIGHTS?: {}".format(param.isnan().any()))
                        print("WEIGHTS?: {}".format(param.max()))
                        print("WEIGHTS?: {}".format(param.min()))
                    print("NANS IN INPUT?: {}".format(x.isnan().any()))
                    print("NANS IN TARGET?: {}".format(yy.isnan().any()))
                    print("NANS IN GRID?: {}".format(yy.isnan().any()))
                    print("LOSS: {}".format(loss))
                    loss.backward()
                    for grad in model.parameters():
                        print("NANS IN GRADS?: {}".format(param.grad.isnan().any()))
                    raise ValueError("\nERROR IN OUTPUT\n")
            elif(isinstance(model,DeepONet1D)):
                x = torch.swapaxes(xx, 1, 2)
                grid = torch.swapaxes(grid.unsqueeze(-1), 1, 2)
                im = model(x, grid)[...,0]
                if(len(yy.shape) == 3 and len(im.shape) == 2):
                    im = im.unsqueeze(-1)
                loss = loss_fn(yy, im)

            # Guarantees we're able to plot at least a few from first batch
            if(bn == 0):
                #y_train_true = yy[:,0,:].clone()
                y_train_true = y.clone()
                y_train_pred = im.clone()

            train_l2_step += loss.item()
            train_l2_full += loss.item()
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_l2s.append(train_l2_full/(bn+1))

        if ep % config['validate'] == 0:
            val_l2_step = 0
            val_l2_full = 0
            with torch.no_grad():
                model.eval()
                #for bn, (xx, yy, grid) in enumerate(val_loader):
                for bn, (xx, y, grid, tokens, t, target) in enumerate(val_loader):

                    # Each model handles input differnetly
                    #if(isinstance(model, (FNO1d, OFormer1D))):
                    if(isinstance(model, FNO1d)):
                        #x = torch.swapaxes(xx, 1, 2)
                        ##grid = torch.swapaxes(grid, 1, 2)
                        #grid = grid.unsqueeze(-1)#torch.swapaxes(grid, 1, 2)
                        ##im, loss = model.get_loss(x, yy[:,0,:], grid, loss_fn)
                        #if(len(yy.shape) == 3):
                        #    yy = yy[...,0]
                        #im, loss = model.get_loss(x, yy, grid, loss_fn)
                        x = torch.swapaxes(xx, 1, 2)
                        grid = torch.swapaxes(grid.unsqueeze(1), 1, 2)
                        t = t.cuda()
                        target = target.cuda()

                        # Standard forward pass
                        x = torch.swapaxes(xx, 1, 2).cuda()
                        im = model(x, grid, t=t, coeffs=target)[...,0]
                        loss = loss_fn(im, y)

                    elif(isinstance(model,DeepONet1D)):
                        x = torch.swapaxes(xx, 1, 2)
                        grid = torch.swapaxes(grid.unsqueeze(-1), 1, 2)
                        im = model(x, grid)[...,0]
                        if(len(yy.shape) == 3 and len(im.shape) == 2):
                            im = im.unsqueeze(-1)
                        loss = loss_fn(yy, im)

                    # Guarantees we're able to plot at least a few from first batch
                    if(bn == 0):
                        #y_val_true = yy[:,0,:].clone()
                        y_val_true = y.clone()
                        y_val_pred = im.clone()

                    val_l2_step += loss.item()
                    val_l2_full += loss.item()
                
                if  val_l2_full < loss_val_min:
                    loss_val_min = val_l2_full
                    best_ep = ep
                    best_model = model.state_dict()
                    best_optimizer = optimizer.state_dict()
                    best_loss_val_min = loss_val_min

                    # Save best
                    torch.save({
                        'epoch': best_ep,
                        'model_state_dict': best_model,
                        'optimizer_state_dict': best_optimizer,
                        'loss': best_loss_val_min
                    }, model_path)

        model.train()
        val_l2s.append(val_l2_full/(bn+1))
                
        t2 = default_timer()
        scheduler.step()
        if(ep%config['log_freq'] == 0):
            print('epoch: {0}, loss: {1:.5f}, time: {2:.5f}s, trainL2: {3:.5f}, testL2: {4:.5f}'\
                .format(ep, loss.item(), t2 - t1, train_l2s[-1], val_l2s[-1]))
            np.save("./{}/train_l2s_{}.npy".format(path, seed), train_l2s)
            np.save("./{}/val_l2s_{}.npy".format(path, seed), val_l2s)

        if(ep%config['progress_plot_freq'] == 0 and len(y_train_true) >= 4):
            progress_plots(ep, y_train_true, y_train_pred, y_val_true, y_val_pred, path, seed=seed)


    # Make sure to capture last
    print('epoch: {0}, loss: {1:.5f}, time: {2:.5f}s, trainL2: {3:.5f}, testL2: {4:.5f}'\
          .format(ep, loss.item(), t2 - t1, train_l2s[-1], val_l2s[-1]))
    np.save("./{}/train_l2s_{}.npy".format(path, seed), train_l2s)
    np.save("./{}/val_l2s_{}.npy".format(path, seed), val_l2s)
    progress_plots(ep, y_train_true, y_train_pred, y_val_true, y_val_pred, path, seed=seed)

    test_vals = []
    #model.eval()
    test_value = evaluate(test_loader, model, loss_fn)
    test_vals.append(test_value)
    print("TEST VALUE FROM LAST EPOCH: {0:5f}".format(test_value))
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    test_value = evaluate(test_loader, model, loss_fn)
    test_vals.append(test_value)
    print("TEST VALUE BEST LAST EPOCH: {0:5f}".format(test_value))
    np.save("./{}/test_vals_{}.npy".format(path, seed), test_vals)
    #rollout_eval(config, model_path, prefix, seed=seed, test_loader=test_loader)
    if(config['train_style'] == 'arbitrary_step'):
        as_rollout_eval(config, path, model_path, prefix, seed=seed, test_loader=test_loader)
    elif(config['train_style'] == 'next_step'):
        ar_rollout_eval(config, path, model_path, prefix, seed=seed, test_loader=test_loader)
    model.train()
            

def single_run_training(old_model, config, prefix, train_loader, val_loader, dset):
    
    ################################################################
    # load data
    ################################################################
    del train_loader
    del val_loader
    torch.cuda.empty_cache()
    
    config['flnm'] = dset
    #config['return_text'] = False
    path = "{}pretrain_{}_{}_{}_{}".format(train_args['results_dir'], config['model_name'], config['contrastive_loss'], config['similarity'], prefix)
    f = h5py.File("{}{}".format(config['base_path'], config['data_name']), 'r')
    model_name = config['flnm'] + '_{}'.format(config['model_name']) + "_{}.pt".format(seed)
    model_path = path + "/" + model_name
    
    print("Filename: {}, Seed: {}\n".format(config['flnm'], config['seed']))
    if(config['forcing_term'] == 'non_td'):
        if(dset == 'Heat'): 
            f = h5py.File("../xwide_non_td_heat_2000.h5", 'r')
        elif(dset == 'Burgers'):
            f = h5py.File("../xwide_non_td_burgers_250.h5", 'r')
        elif(dset == 'KdV'):
            f = h5py.File("../xwide_non_td_kdv_250.h5", 'r')
    elif(config['forcing_term'] == 'none'):
        if(dset == 'Heat'): 
            f = h5py.File("/home/cooper/new_long_xwide_no_forcing_heat_2000.h5", 'r')
            #f = h5py.File("/home/cooper/finetune_new_long_xwide_no_forcing_heat_2000.h5", 'r')
        elif(dset == 'Burgers'):
            f = h5py.File("/home/cooper/new_long_xwide_no_forcing_burgers_250.h5", 'r')
            #f = h5py.File("/home/cooper/finetune_new_long_xwide_no_forcing_burgers_250.h5", 'r')
        elif(dset == 'Advection'):
            f = h5py.File("/home/cooper/new_long_xwide_no_forcing_advection_2000.h5", 'r')
            #f = h5py.File("/home/cooper/finetune_new_long_xwide_no_forcing_advection_2000.h5", 'r')
        #elif(dset == 'KdV'):
        #    f = h5py.File("../new_long_xwide_no_forcing_kdv_500.h5", 'r')
        else:
            raise ValueError("Invalid choice of data set.")

    #train_loader, val_loader, test_loader = get_data(f, config)
    # Reset batch size
    train_loader, val_loader, test_loader = get_single_data(f, config)
    model.pretrain_off()
    
    ################################################################
    # training and evaluation
    ################################################################
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')
    # Use Adam for Hyena
    #optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    #if(isinstance(model, OFormer1D)):
    #    print("\nUSING ONECYCLELER SCHEDULER\n")
    #    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'],# div_factor=1e6,
    #                                                    steps_per_epoch=len(train_loader), epochs=config['epochs'])
    #else:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step'], gamma=config['scheduler_gamma'])
    
    loss_fn = nn.L1Loss(reduction="mean")
    #loss_fn = nn.MSELoss(reduction="mean")
    loss_val_min = np.infty
    
    start_epoch = 0
    
    train_l2s, val_l2s = [], []
    for ep in tqdm(range(start_epoch, config['epochs'])):
        model.train()
        #model.eval()
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        #for bn, (xx, yy, grid) in enumerate(train_loader):
        for bn, (xx, y, grid, tokens, t, target) in enumerate(train_loader):
            
            # Each model handles input differnetly
            #if(isinstance(model, (FNO1d, OFormer1D))):
            if(isinstance(model, FNO1d)):
                ##print(xx.shape, grid.shape)
                #x = torch.swapaxes(xx, 1, 2)
                #grid = grid.unsqueeze(-1)#torch.swapaxes(grid, 1, 2)
                ##im = model(x, grid)
                ##print(yy.shape)
                ##im, loss = model.get_loss(x, yy[:,0,:], grid, loss_fn)
                ##print(x.shape, yy.shape, grid.shape)
                #im, loss = model.get_loss(x, yy, grid, loss_fn)
                ##raise
                x = torch.swapaxes(xx, 1, 2)
                grid = torch.swapaxes(grid.unsqueeze(1), 1, 2)
                t = t.cuda()
                target = target.cuda()

                # Standard forward pass
                x = torch.swapaxes(xx, 1, 2).cuda()
                im = model(x, grid, t=t, coeffs=target)[...,0]
                loss = loss_fn(im, y)
            elif(isinstance(model,DeepONet1D)):
                x = torch.swapaxes(xx, 1, 2)
                grid = torch.swapaxes(grid.unsqueeze(-1), 1, 2)
                im = model(x, grid)[...,0]
                #print(im.shape)
                #print(yy.shape)
                loss = loss_fn(yy, im)

            # Guarantees we're able to plot at least a few from first batch
            if(bn == 0):
                #y_train_true = yy[:,0,:].clone()
                y_train_true = y.clone()
                y_train_pred = im.clone()

            train_l2_step += loss.item()
            train_l2_full += loss.item()
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #if(isinstance(model, OFormer1D)):
            #    scheduler.step()

        train_l2s.append(train_l2_full/(bn+1))

        if ep % config['validate'] == 0:
            val_l2_step = 0
            val_l2_full = 0
            with torch.no_grad():
                model.eval()
                #for bn, (xx, yy, grid) in enumerate(val_loader):
                for bn, (xx, y, grid, tokens, t, target) in enumerate(val_loader):

                    # Each model handles input differnetly
                    #if(isinstance(model, (FNO1d, OFormer1D))):
                    if(isinstance(model, FNO1d)):
                        #x = torch.swapaxes(xx, 1, 2)
                        ##grid = torch.swapaxes(grid, 1, 2)
                        #grid = grid.unsqueeze(-1)#torch.swapaxes(grid, 1, 2)
                        ##im, loss = model.get_loss(x, yy[:,0,:], grid, loss_fn)
                        #im, loss = model.get_loss(x, yy, grid, loss_fn)
                        x = torch.swapaxes(xx, 1, 2)
                        grid = torch.swapaxes(grid.unsqueeze(1), 1, 2)
                        t = t.cuda()
                        target = target.cuda()

                        # Standard forward pass
                        x = torch.swapaxes(xx, 1, 2).cuda()
                        im = model(x, grid, t=t, coeffs=target)[...,0]
                        loss = loss_fn(im, y)
                    elif(isinstance(model,DeepONet1D)):
                        x = torch.swapaxes(xx, 1, 2)
                        grid = torch.swapaxes(grid.unsqueeze(-1), 1, 2)
                        im = model(x, grid)[...,0]
                        loss = loss_fn(yy, im)

                    # Guarantees we're able to plot at least a few from first batch
                    if(bn == 0):
                        #y_val_true = yy[:,0,:].clone()
                        y_val_true = y.clone()
                        y_val_pred = im.clone()

                    val_l2_step += loss.item()
                    val_l2_full += loss.item()
                
                if  val_l2_full < loss_val_min:
                    loss_val_min = val_l2_full
                    best_ep = ep
                    best_model = model.state_dict()
                    best_optimizer = optimizer.state_dict()
                    best_loss_val_min = loss_val_min

                    # Save best
                    torch.save({
                        'epoch': best_ep,
                        'model_state_dict': best_model,
                        'optimizer_state_dict': best_optimizer,
                        'loss': best_loss_val_min
                    }, model_path)

        model.train()
        val_l2s.append(val_l2_full/(bn+1))
                
        t2 = default_timer()
        #if(not isinstance(model, OFormer1D)):
        scheduler.step()

        if(ep%config['log_freq'] == 0):
            print('epoch: {0}, loss: {1:.5f}, time: {2:.5f}s, trainL2: {3:.5f}, testL2: {4:.5f}'\
                .format(ep, loss.item(), t2 - t1, train_l2s[-1], val_l2s[-1]))
            np.save("./{}/{}_train_l2s_{}.npy".format(path, dset, seed), train_l2s)
            np.save("./{}/{}_val_l2s_{}.npy".format(path, dset, seed), val_l2s)

        if(ep%config['progress_plot_freq'] == 0 and len(y_train_true) >= 4):
            progress_plots(ep, y_train_true, y_train_pred, y_val_true, y_val_pred, path, seed=seed)


    # Make sure to capture last
    print('epoch: {0}, loss: {1:.5f}, time: {2:.5f}s, trainL2: {3:.5f}, testL2: {4:.5f}'\
          .format(ep, loss.item(), t2 - t1, train_l2s[-1], val_l2s[-1]))
    np.save("./{}/{}_train_l2s_{}.npy".format(path, dset, seed), train_l2s)
    np.save("./{}/{}_val_l2s_{}.npy".format(path, dset, seed), val_l2s)
    progress_plots(ep, y_train_true, y_train_pred, y_val_true, y_val_pred, path, seed=seed, dset=dset)

    test_vals = []
    test_value = evaluate(test_loader, model, loss_fn)
    test_vals.append(test_value)
    print("TEST VALUE FROM LAST EPOCH: {0:5f}".format(test_value))
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    test_value = evaluate(test_loader, model, loss_fn)
    test_vals.append(test_value)
    print("TEST VALUE BEST LAST EPOCH: {0:5f}".format(test_value))
    np.save("./{}/{}_test_vals_{}.npy".format(path, dset, seed), test_vals)
    if(config['train_style'] == 'arbitrary_step'):
        as_rollout_eval(config, path, model_path, prefix, seed=seed, test_loader=test_loader)
    elif(config['train_style'] == 'next_step'):
        ar_rollout_eval(config, path, model_path, prefix, seed=seed, test_loader=test_loader)
    model.train()


def freeze_encoder(model):
    if(isinstance(model, OFormer1D)):
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif(isinstance(model, FNO1d)):
        for param in model.conv0.parameters():
            param.requires_grad = False
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.conv2.parameters():
            param.requires_grad = False
        #for param in model.conv3.parameters():
        #    param.requires_grad = False
        for param in model.w0.parameters():
            param.requires_grad = False
        for param in model.w1.parameters():
            param.requires_grad = False
        for param in model.w2.parameters():
            param.requires_grad = False
        #for param in model.w3.parameters():
        #    param.requires_grad = False
    return model
            

if __name__ == "__main__":
    #raise
    try:
        model_name = sys.argv[1]
    except IndexError:
        print("Default model is FNO. Training FNO.")
        model_name = "fno"
    try:
        assert model_name in ['fno', 'oformer','deeponet']
    except AssertionError as e:
        print("\nModel must be one of: fno, oformer or deeponet. Model selected was: {}\n".format(model_name))
        raise
    #if(model_name == 'oformer'):
    #    raise

    # Load config
    with open("./{}_config.yaml".format(model_name), 'r') as stream:
        config = yaml.safe_load(stream)

    # Get arguments and get rid of unnecessary ones
    train_args = config['args']
    train_args['model_name'] = model_name
    device = train_args['device']
    prefix = train_args['flnm'] + "_" + train_args['train_style']
    train_args['results_dir'] = train_args['results_dir'] + str(train_args['num_samples']) + "_" + str(train_args['pretraining_num_samples']) + "/"
    print("RESULTS_DIR: {}".format(train_args['results_dir']))
    print("PREFIX: {}".format(prefix))
    os.makedirs("{}pretrain_{}_{}_{}_{}".format(train_args['results_dir'], model_name, train_args['contrastive_loss'],
                                                train_args['similarity'], prefix),
                                             exist_ok=True)
    shutil.copy("./{}_config.yaml".format(model_name),
                "{}pretrain_{}_{}_{}_{}/{}_config.yaml".format(train_args['results_dir'], model_name, train_args['contrastive_loss'],
                                                            train_args['similarity'], prefix, model_name))
    shutil.copy("./finetune_plot_progress.py", "{}pretrain_{}_{}_{}_{}/finetine_plot_progress.py".format(train_args['results_dir'], model_name,
                                                            train_args['contrastive_loss'], train_args['similarity'], prefix))
    shutil.copy("./pretrain_plot_progress.py", "{}pretrain_{}_{}_{}_{}/pretrain_plot_progress.py".format(train_args['results_dir'],
                model_name, train_args['contrastive_loss'], train_args['similarity'], prefix))
    shutil.copy("./compare_results.py", "{}/compare_progress.py".format(train_args['results_dir']))

    for seed in range(train_args.pop('num_seeds')):
    #for seed in [0,1,2,3]:
    #for seed in [2,3]:
    #for seed in [4]:
    #for seed in [0]:
    #for seed in [1]:
    #for seed in [2]:
    #for seed in [3]:
    #for seed in [4]:
        #if(seed in [0]):
        #    continue
        print("\nSEED: {}\n".format(seed))
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_args['seed'] = seed

        model = get_model(model_name, train_args)
        if(hasattr(model, "dropout")):
            print(model.dropout)
        train_args['return_text'] = True
        #train_args['flnm'] = 'all'
        #model._finetune = True
        if(train_args['contrastive_loss'] != 'none'):
            if(train_args['load_pretrained']):
                try:
                    model, train_loader, val_loader, test_loader, model_path = load_model(model, train_args, prefix, seed=seed)
                except FileNotFoundError:
                    print("NO PRETRAINED MODEL FOUND. RUNNING PRETRAINING.")
                    model, train_loader, val_loader, test_loader, model_path = pretraining(model, train_args, prefix)
            else:
                model, train_loader, val_loader, test_loader, model_path = pretraining(model, train_args, prefix)

            #model = freeze_encoder(model)
        model._finetune = False

        #train_args['dropout'] = train_args['finetune_dropout']
        model = get_model(model_name, train_args, finetune=True)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        if(hasattr(model, "dropout")):
            print(model.dropout)
        #run_training(model, train_args, prefix, train_loader, val_loader, test_loader)

        for dset in ['Heat', 'Burgers', 'Advection']:
        #for dset in ['Heat', 'Burgers']:
        #for dset in ['Advection']:

            print(model_path)
            #model = get_model(model_name, train_args, finetune=True)
            model = get_model(model_name, train_args, finetune=True)
            model.load_state_dict(torch.load(model_path)['model_state_dict'])
            model._finetune = True
            #model = freeze_encoder(model)

            single_run_training(model, train_args.copy(), prefix, train_loader, val_loader, dset)
    print("Done.")


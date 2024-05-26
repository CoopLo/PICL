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
from utils import LpLoss

import yaml
from tqdm import tqdm
import h5py
from matplotlib import pyplot as plt

DEBUG = True

def progress_plots(ep, y_train_true, y_train_pred, y_val_true, y_val_pred, path="progress_plots", seed=None):
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


def get_model(model_name, config):
    if(model_name == "fno"):
        model = FNO1d(config['num_channels'], config['modes'], config['width'], config['initial_step']+4, config['finetune_dropout'])
    elif(model_name == "oformer"):
        encoder = Encoder1D(input_channels=config['input_channels'], in_emb_dim=config['in_emb_dim'],
                            out_seq_emb_dim=config['out_seq_emb_dim'], depth=config['depth'], dropout=config['finetune_dropout'],
                            res=config['enc_res'])
        decoder = STDecoder1D(latent_channels=config['latent_channels'], out_channels=config['out_channels'],
                                     decoding_depth=config['decoding_depth'], scale=config['scale'], res=config['dec_res'])
        model = OFormer1D(encoder, decoder)
    elif(model_name == "deeponet"):
        model = DeepONet1D(config['branch_net'], config['trunk_net'], config['activation'], config['kernel_initializer'])
    
    model.to(device)
    return model


def get_data(f, config):
    print(config['flnm'])
    if(config['flnm'] == 'all'):
        train_data = TransformerMultiOperatorDataset(config['base_path'],
                                split="train",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['num_samples'],
                                train_style=config['train_style'],
                                rollout_length=config['rollout_length'],
                                seed=config['seed'],
                                forcing_term=config['forcing_term'],
                                finetune=True,
                                debug=DEBUG,
        )
        train_data.data = train_data.data.to(device)
        train_data.grid = train_data.grid.to(device)
        val_data = TransformerMultiOperatorDataset(config['base_path'],
                                split="val",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['num_samples'],
                                train_style=config['train_style'],
                                rollout_length=config['rollout_length'],
                                seed=config['seed'],
                                forcing_term=config['forcing_term'],
                                finetune=True,
                                debug=DEBUG,
        )
        val_data.data = val_data.data.to(device)
        val_data.grid = val_data.grid.to(device)
        test_data = TransformerMultiOperatorDataset(config['base_path'],
                                split="test",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['num_samples'],
                                train_style=config['train_style'],
                                rollout_length=config['rollout_length'],
                                seed=config['seed'],
                                forcing_term=config['forcing_term'],
                                finetune=True,
                                debug=DEBUG,
        )
        test_data.data = test_data.data.to(device)
        test_data.grid = test_data.grid.to(device)
    else:
        train_data = TransformerOperatorDataset(f, config['flnm'],
                                split="train",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['num_samples'],
                                train_style=config['train_style'],
                                rollout_length=config['rollout_length'],
                                seed=config['seed'],
                                debug=DEBUG,
        )
        train_data.data = train_data.data.to(device)
        train_data.grid = train_data.grid.to(device)

        val_name = config['data_name'].replace('finetune', 'validate')

        # Get new name based on split
        if('heat' in config['data_name']):
            val_name = "validate_new_long_xwide_no_forcing_heat_100.h5"
        elif('burgers' in config['data_name']):
            val_name = "validate_new_long_xwide_no_forcing_burgers_25.h5"
        elif('advection' in config['data_name']):
            val_name = "validate_new_long_xwide_no_forcing_advection_125.h5"
        f = h5py.File("{}{}".format(config['base_path'], val_name), 'r')
        val_data = TransformerOperatorDataset(f, config['flnm'],
                                split="val",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['num_samples'],
                                train_style=config['train_style'],
                                rollout_length=config['rollout_length'],
                                seed=config['seed'],
                                debug=DEBUG,
        )
        val_data.data = val_data.data.to(device)
        val_data.grid = val_data.grid.to(device)

        # Get new name based on split
        if('heat' in config['data_name']):
            test_name = "test_new_long_xwide_no_forcing_heat_200.h5"
        elif('burgers' in config['data_name']):
            test_name = "test_new_long_xwide_no_forcing_burgers_50.h5"
        elif('advection' in config['data_name']):
            test_name = "test_new_long_xwide_no_forcing_advection_250.h5"
        f = h5py.File("{}{}".format(config['base_path'], test_name), 'r')
        test_data = TransformerOperatorDataset(f, config['flnm'],
                                split="test",
                                initial_step=config['initial_step'],
                                reduced_resolution=config['reduced_resolution'],
                                reduced_resolution_t=config['reduced_resolution_t'],
                                reduced_batch=config['reduced_batch'],
                                saved_folder=config['base_path'],
                                num_t=config['num_t'],
                                num_x=config['num_x'],
                                sim_time=config['sim_time'],
                                num_samples=config['num_samples'],
                                train_style=config['train_style'],
                                rollout_length=config['rollout_length'],
                                seed=config['seed'],
                                debug=DEBUG,
        )
        test_data.data = test_data.data.to(device)
        test_data.grid = test_data.grid.to(device)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config['batch_size'],
                                               num_workers=config['num_workers'], shuffle=True)#,
                                               #generator=torch.Generator(device='cuda'))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config['batch_size'],
                                             num_workers=config['num_workers'], shuffle=False)#,
                                             #generator=torch.Generator(device='cuda'))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'],
                                             num_workers=config['num_workers'], shuffle=False)#,
                                             #generator=torch.Generator(device='cuda'))

    #assert not (bool(set(train_data.data_list) & \
    #                 set(val_data.data_list)) | \
    #            bool(set(train_data.data_list) & \
    #                 set(test_data.data_list)) & \
    #            bool(set(val_data.data_list) & \
    #                 set(test_data.data_list)))
    #TODO Check data at least once...

    return train_loader, val_loader, test_loader


def get_loss(model, xx, y, grid, t, target, loss_fn):
    if(isinstance(model, FNO1d)):
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

    return im, loss



def evaluate(test_loader, model, loss_fn):
    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        model.eval()
        #for bn, (xx, yy, grid) in enumerate(test_loader):
        for bn, (xx, y, grid, t, target) in enumerate(test_loader):
            
            im, loss = get_loss(model, xx, y, grid, t, target, loss_fn)
            if(loss == torch.inf):
                print(im.shape)
                print(y.shape)
                for i in range(im.shape[0]):
                    if(loss_fn(im[i], y[i]) == torch.inf):
                        fig, ax = plt.subplots()
                        ax.plot(y[i].cpu(), label="Ground Truth")
                        ax.plot(im[i].cpu(), label="Prediction")
                        ax.legend(loc='best')

                        for idx, d in enumerate(test_loader.dataset.data_list):
                            #print(test_loader.dataset.data[idx].shape)
                            if((test_loader.dataset.data[idx] == torch.zeros((50,50)).cuda()).all()):
                                print(test_loader.dataset.data_list[idx])
                        raise
                        plt.show()

                    #print(y[i])
                raise

            test_l2_step += loss.item()
            test_l2_full += loss.item()
    #raise
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


def run_training(model, config, prefix):
    
    ################################################################
    # load data
    ################################################################
    
    path = "{}{}_{}".format(train_args['results_dir'], config['model_name'], prefix)
    f = h5py.File("{}{}".format(config['base_path'], config['data_name']), 'r')
    model_name = config['flnm'] + '_{}'.format(config['model_name']) + "_{}.pt".format(seed)
    model_path = path + "/" + model_name
    
    print("Filename: {}, Seed: {}\n".format(config['flnm'], config['seed']))

    train_loader, val_loader, test_loader = get_data(f, config)
    #train_loader.dataset.pretrain()
    #val_loader.dataset.pretrain()
    #test_loader.dataset.pretrain()
    
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
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'],# div_factor=1e6,
    #                                                steps_per_epoch=len(train_loader), epochs=config['epochs'])
    
    loss_fn = nn.L1Loss(reduction="mean")
    #loss_fn = nn.MSELoss(reduction="mean")
    loss_val_min = np.infty
    
    start_epoch = 0
    
    train_l2s, val_l2s = [], []
    for ep in tqdm(range(start_epoch, config['epochs'])):
        #model.train()
        model.eval()
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        #for bn, (xx, yy, grid) in enumerate(train_loader):
        for bn, (xx, y, grid, t, target) in enumerate(train_loader):
            
            im, loss = get_loss(model, xx, y, grid, t, target, loss_fn)

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
            if(isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)):
                scheduler.step()


        train_l2s.append(train_l2_full/(bn+1))

        if ep % config['validate'] == 0:
            val_l2_step = 0
            val_l2_full = 0
            with torch.no_grad():
                model.eval()
                for bn, (xx, y, grid, t, target) in enumerate(val_loader):

                    im, loss = get_loss(model, xx, y, grid, t, target, loss_fn)

                    # Guarantees we're able to plot at least a few from first batch
                    if(bn == 0):
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
        if(isinstance(scheduler, torch.optim.lr_scheduler.StepLR)):
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

    # Evaluate using LpLoss
    loss_fn = LpLoss(d=1)
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
    if(config['train_style'] == 'arbitrary_step'):
        as_rollout_eval(config, path, model_path, prefix, seed=seed, test_loader=test_loader)
    elif(config['train_style'] == 'next_step'):
        ar_rollout_eval(config, path, model_path, prefix, seed=seed, test_loader=test_loader)
    model.train()
            
if __name__ == "__main__":
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

    with open("./{}_config.yaml".format(model_name), 'r') as stream:
        config = yaml.safe_load(stream)
    train_args = config['args']
    train_style = train_args['train_style']
    NUM_SAMPLES = 500
    #pns = 1000
    #for ns in [100, 500, 1000, 5000]:
    for ns in [100, 500, 1000]:
    #for ns in [50, 100, 500]:
    #for ns in [1000]:#, 500, 100]:
    #for ns in [100]:
    #for ns in [50]:
        for vs in [
                   ['Heat', 'finetune_new_long_xwide_no_forcing_heat_1500.h5'],
                   ['Burgers', 'finetune_new_long_xwide_no_forcing_burgers_250.h5'],
                   ['Advection', 'finetune_new_long_xwide_no_forcing_advection_2000.h5'],
                   ['all', 'new_long_xwide_no_forcing_advection_2000.h5'],
                  ]:
            # Load config
            with open("./{}_config.yaml".format(model_name), 'r') as stream:
                config = yaml.safe_load(stream)
            train_args = config['args']
            train_args['train_style'] = train_style
            train_args['num_samples'] = ns
            #train_args['pretraining_num_samples'] = pns

            # Update file and data name
            train_args['flnm'] = vs[0]
            train_args['data_name'] = vs[1]
            #train_args['train_style'] = 'next_step'

            # Get arguments and get rid of unnecessary ones
            train_args['model_name'] = model_name
            device = train_args['device']
            #prefix = train_args['flnm'] + "_" + train_args['data_name'].split("_")[0] + "_" + train_args['train_style']
            prefix = train_args['flnm'] + "_" + train_args['train_style']
            train_args['results_dir'] = train_args['results_dir'] + str(train_args['num_samples']) + "/"
            print("RESULTS_DIR: {}".format(train_args['results_dir']))
            print("PREFIX: {}".format(prefix))
            #os.makedirs("{}pretrain_{}_{}".format(train_args['results_dir'], model_name, prefix), exist_ok=True)
            os.makedirs("{}{}_{}".format(train_args['results_dir'], model_name, prefix), exist_ok=True)
            shutil.copy("./{}_config.yaml".format(model_name),
                        "{}{}_{}/{}_config.yaml".format(train_args['results_dir'], model_name, prefix, model_name))
            shutil.copy("./plot_progress.py", "{}{}_{}/plot_progress.py".format(train_args['results_dir'], model_name, prefix))
            shutil.copy("./compare_results.py", "{}/compare_progress.py".format(train_args['results_dir']))


            #for seed in range(train_args.pop('num_seeds')):
            for seed in range(train_args['num_seeds']):
            #for seed in [0]:
            #for seed in [1]:
            #for seed in [2]:
            #for seed in [3]:
            #for seed in [4]:
                #if(seed != 0):
                #    continue
                #if(seed not in [3,4]):
                #    continue
                print("\nSEED: {}\n".format(seed))
                torch.manual_seed(seed)
                np.random.seed(seed)
                train_args['seed'] = seed

                model = get_model(model_name, train_args)
                run_training(model, train_args, prefix)
            print("Done.")

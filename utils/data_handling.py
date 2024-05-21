# -*- coding: utf-8 -*-
"""
       <NAME OF THE PROGRAM THIS FILE BELONGS TO>

  File:     utils.py
  Authors:  Timothy Praditia (timothy.praditia@iws.uni-stuttgart.de)
            Raphael Leiteritz (raphael.leiteritz@ipvs.uni-stuttgart.de)
            Makoto Takamoto (makoto.takamoto@neclab.eu)
            Francesco Alesiani (makoto.takamoto@neclab.eu)

NEC Laboratories Europe GmbH, Copyright (c) <year>, All rights reserved.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

       PROPRIETARY INFORMATION ---

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor.

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""

import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader
import os
import glob
import h5py
import numpy as np
import math as mt
import time
from tqdm import tqdm
import itertools
import random
import copy
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'


class TransformerOperatorDataset(Dataset):
    def __init__(self, f, filename,
                 initial_step=10,
                 saved_folder='./data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 num_t=200,
                 num_x=200,
                 sim_time=-1,
                 split="train",
                 test_ratio=0.2,
                 val_ratio=0.2,
                 num_samples=None,
                 return_text=False,
                 rollout_length=10,
                 train_style='fixed_future',
                 ssl=False, forcing=False, seed=0,
                 ):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """
        
        # Define path to files
        self.file_path = os.path.abspath(f.filename)
        #self.file_path = os.path.abspath(saved_folder + filename + ".h5")
        self.return_text = return_text
        self.train_style = train_style
        self.ssl = ssl
        self.forcing = forcing
        self._pretrain = False
        num_t = 200
        self.reduced_resolution = reduced_resolution
        self.reduced_resolution_t = reduced_resolution_t
        
        # Extract list of seeds
        print("\nSEED: {}".format(seed))
        np.random.seed(seed)
        if(filename != "all"):
            data_list = []
            #for key in f.keys():
            #    if(filename in key):
            #        data_list.append(key)
            if('Heat' in filename):
                print("\nHERE\n")
                data_list = [key for key in f.keys() if(len(key.split("_")) == 3)]
            else:
                data_list = [key for key in f.keys()]
            np.random.shuffle(data_list)
        else:
            #data_list = list([key for key in f.keys() if("KdV" not in key))
            data_list = [key for key in f.keys()]
            np.random.shuffle(data_list)

        self.data_list = data_list

        # Get target split. Seeding is required to make this reproducible.
        # This splits each run, lets try a better shuffle
        if(num_samples is not None):
            data_list = data_list[:num_samples]
        train_idx = int(len(data_list) * (1 - test_ratio - val_ratio))
        val_idx = int(len(data_list) * (1-test_ratio))
        #print(train_idx, val_idx)
        #raise

        # Make sure no data points occur in two splits
        assert not (bool(set(self.data_list[:train_idx]) & \
                         set(self.data_list[train_idx:val_idx])) | \
                    bool(set(self.data_list[val_idx:]) & \
                         set(self.data_list[train_idx:])) & \
                    bool(set(self.data_list[val_idx:]) & \
                         set(self.data_list[train_idx:val_idx])))

        if(split == "train"):
            self.data_list = np.array(data_list[:train_idx])
        elif(split == "val"):
            self.data_list = np.array(data_list[train_idx:val_idx])
        elif(split == "test"):
            self.data_list = np.array(data_list[val_idx:])
        else:
            raise ValueError("Select train, val, or test split. {} is invalid.".format(split))
        #print(self.data_list)
        #raise
        
        # Time steps used as initial conditions
        self.initial_step = initial_step
        self.rollout_length = rollout_length

        self.WORDS = ['(', ')', '+', '-', '*', '/', 'Derivative', 'Sum', 'j', 'A_j', 'l_j',
                 'omega_j', 'phi_j'    , 'sin', 't', 'u', 'x', 'dirichlet', 'neumann',
                 "None", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10^',
                 'E', 'e', ',', '.', '&']
        self.word2id = {w: i for i, w in enumerate(self.WORDS)}
        self.id2word = {i: w for i, w in enumerate(self.WORDS)}
        self.num_t = num_t
        self.num_x = num_x
        if('kdv' in f.filename):
            self.num_x = 400
        self.name = "pde_{}-{}".format(self.num_t, self.num_x)

        self.h5_file = h5py.File(self.file_path, 'r')
        #self.sim_time = sim_time
        if(self.train_style == 'next_step' and sim_time == -1):
            self.sim_time = num_t//self.reduced_resolution_t - 1
        elif(self.train_style == 'arbitrary_step' and sim_time == -1):
            self.sim_time = num_t//self.reduced_resolution_t - 1
        else:
            self.sim_time = sim_time

        self.data = []
        self.grid = []
        self.time = []
        self.tokens = []
        self.available_idxs = []
        self.all_operator_maps = []
        #print(len(self.data_list))
        #raise
        print("Gathering data...")
        for i in tqdm(range(len(self.data_list))):
            seed_group = self.h5_file[self.data_list[i]]
            if('kdv' in f.filename):
                self.data.append(seed_group[self.name][0][:,::2][::self.reduced_resolution_t,::self.reduced_resolution])
            else:
                self.data.append(seed_group[self.name][0][::self.reduced_resolution_t,::self.reduced_resolution])

            if(self.train_style == 'next_step'):
                idxs = np.arange(0, len(seed_group[self.name][0])//self.reduced_resolution_t)[self.initial_step:self.sim_time]
            elif(self.train_style == 'arbitrary_step'):
                idxs = np.arange(0, len(seed_group[self.name][0])//self.reduced_resolution_t)[self.initial_step:self.sim_time]
                #idxs = np.arange(0, len(seed_group[self.name][0]))[self.initial_step:]
            
            elif(self.train_style == 'rollout'):
                length = len(seed_group[self.name][0])
                idxs = np.arange(0, length)[self.initial_step:length-self.rollout_length]
            elif(self.train_style == 'fixed_future'):
                idxs = np.array([i])

            if(len(self.available_idxs) != 0 and self.train_style != 'fixed_future'):
                # Needs to make sure it wraps all the way back around...
                #TODO Make sure this is right
                #idxs += self.available_idxs[-1] + 1 if(self.train_style == 'next_step') else \
                #        self.available_idxs[-1] + 1 + self.rollout_length if(self.train_style == 'rollout') else \
                #                                self.available_idxs[-1] + 100 - self.sim_time#self.available_idxs[-1] + 1
                idxs += self.num_t//self.reduced_resolution_t - (self.sim_time-1) + self.available_idxs[-1] \
                        if(self.train_style in ['next_step', 'arbitrary_step']) else \
                        self.available_idxs[-1] + 1 + self.rollout_length if(self.train_style == 'rollout') else \
                        self.available_idxs[-1] + len(seed_group[self.name][0]) - self.sim_time + 1
                #idxs += self.available_idxs[-1] + 1 if(self.train_style == 'next_step') else \
                #        self.available_idxs[-1] + 1 + self.rollout_length if(self.train_style == 'rollout') else \
                #                                self.available_idxs[-1] + 1
            self.available_idxs.extend(idxs)

            if('kdv' in f.filename):
                self.grid.append(np.array(seed_group[self.name].attrs["x"][::2][::self.reduced_resolution], dtype='f'))
            else:
                self.grid.append(np.array(seed_group[self.name].attrs["x"][::self.reduced_resolution], dtype='f'))
            if(self.return_text):
                #self.tokens.append(list(torch.Tensor(seed_group[self.name].attrs['encoded_tokens'])))
                self.tokens.append(torch.Tensor(seed_group[self.name].attrs['encoded_tokens']))
                self.time.append(seed_group[self.name].attrs['t'][::self.reduced_resolution_t])

                dl_split = self.data_list[i].split("_")
                if(dl_split[0] == 'Burgers'):
                    omap = [float(dl_split[2]), float(dl_split[3]), 0]
                    #print(dl_split)
                elif(dl_split[0] == 'Heat'):
                    omap = [0, float(dl_split[2]), 0]
                    #omap = [0, float(dl_split[3]), 0]
                    #print(dl_split)
                elif(dl_split[0] == 'KdV'):
                    omap = [float(dl_split[2]), 0, float(dl_split[3])]
                elif(dl_split[0] == 'Advection'):
                    omap = [0, 0, float(dl_split[2])]
                else:
                    raise ValueError("Invalid 1D data set used. Only Heat, Burgers, and KdV are currently supported.")
                self.all_operator_maps.append(omap)

        self.data = torch.Tensor(np.array(self.data)).to(device=device)#, dtype=torch.float).cuda()
        self.grid = torch.Tensor(np.array(self.grid)).to(device=device)#.cuda()
        self.all_operator_maps = torch.Tensor(np.array(self.all_operator_maps)).to(device=device)
        self.h5_file.close()
        #print(self.available_idxs)
        #raise

        print("\nNUMBER OF SAMPLES: {}".format(len(self.available_idxs)))

        def forcing_term(x, t, As, ls, phis, omegas):
            return np.sum(As[i]*torch.sin(2*np.pi/16. * ls[i]*x + omegas[i]*t + phis[i]) for i in range(len(As)))
        
        # Not suitable for autoregressive training
        if(self.train_style == 'fixed_future'):
            self.all_tokens = torch.empty(len(self.data), 500)#.to(device=device)#.cuda()
            for idx, token in tqdm(enumerate(self.tokens)):
                if(self.return_text):
                    # Encode time token
                    slice_tokens = self._encode_tokens("&" + str(self.time[idx][self.sim_time]))
                    return_tokens = torch.Tensor(self.tokens[idx].clone()).cuda()
                    return_tokens = torch.cat((return_tokens, torch.Tensor(slice_tokens).cuda()))
                    return_tokens = torch.cat((return_tokens, torch.Tensor([len(self.WORDS)]*(500 - len(return_tokens))).cuda()))
                    self.all_tokens[idx] = return_tokens.to(device=device)#.cuda()

        elif(self.train_style in ['next_step', 'arbitrary_step'] and self.return_text):
            # Create array of all legal encodings, pdes, and data
            self.all_tokens = torch.empty(len(self.available_idxs), 500).to(device=device)#.cuda()

            if(self.forcing):
                self.forcing_terms = []
                self.times = torch.empty(len(self.available_idxs))

            print("Processing data...")
            #print(self.available_idxs)
            #print(self.data.shape)
            for idx, sim_idx in tqdm(enumerate(self.available_idxs)):
                #sim_idx = self.available_idxs[idx]      # Get valid prestored index
                sim_num = sim_idx // self.data.shape[1] # Get simulation number
                sim_time = sim_idx % self.data.shape[1] # Get time from that simulation
                #print(sim_num, sim_time)
                if(self.return_text):
                    slice_tokens = self._encode_tokens("&" + str(self.time[sim_num][sim_time]))
                    return_tokens = torch.Tensor(self.tokens[sim_num].clone())

                    # TODO: Maybe put this back
                    #return_tokens = torch.cat((return_tokens, torch.Tensor(slice_tokens).cpu())).cpu()
                    return_tokens = torch.cat((return_tokens, torch.Tensor(slice_tokens))).cuda()

                    return_tokens = torch.cat((return_tokens, torch.Tensor([len(self.WORDS)]*(500 - len(return_tokens))).cuda()))
                    self.all_tokens[idx] = return_tokens.to(device=device)#.cuda()
            #for idx, sim_idx in tqdm(enumerate(self.available_idxs)):
            #    #sim_idx = self.available_idxs[idx]      # Get valid prestored index
            #    sim_num = sim_idx // self.data.shape[1] # Get simulation number
            #    sim_time = sim_idx % self.data.shape[1] # Get time from that simulation
            #    print(sim_num, sim_time)
            #    if(self.return_text):

            #        slice_tokens = self._encode_tokens("&" + str(self.time[sim_num][sim_time]))
            #        return_tokens = torch.Tensor(self.tokens[sim_num].clone())

            #        # TODO: Maybe put this back
            #        #return_tokens = torch.cat((return_tokens, torch.Tensor(slice_tokens).cpu())).cpu()
            #        return_tokens = torch.cat((return_tokens, torch.Tensor(slice_tokens)))
            #        
            #        return_tokens = torch.cat((return_tokens, torch.Tensor([len(self.WORDS)]*(500 - len(return_tokens))).cuda()))
            #        self.all_tokens[idx] = return_tokens.to(device=device)#.cuda()


        if(self.return_text):
            self.all_tokens = self.all_tokens.to(device=device)#.cuda()
        self.time = torch.Tensor(self.time).to(device=device)
        self.data = self.data.cuda()
        self.grid = self.grid.cuda()

    def _encode_tokens(self, all_tokens):
        encoded_tokens = []
        num_concat = 0
        for i in range(len(all_tokens)):
            try: # All the operators, bcs, regular symbols
                encoded_tokens.append(self.word2id[all_tokens[i]])
                if(all_tokens[i] == "&"): # 5 concatenations before we get to lists of sampled values
                    num_concat += 1
            except KeyError: # Numerical values
                if(isinstance(all_tokens[i], str)):
                    for v in all_tokens[i]:
                        try:
                            encoded_tokens.append(self.word2id[v])
                        except KeyError:
                            print(all_tokens)
                            raise
                    if(num_concat >= 5): # We're in a list of sampled parameters
                        encoded_tokens.append(self.word2id[","])
                else:
                    raise KeyError("Unrecognized token: {}".format(all_tokens[i]))
    
        return encoded_tokens

    def __len__(self):
        if(self.train_style == 'fixed_future'):
            return len(self.data_list)
        elif(self.train_style in ['next_step', 'arbitrary_step']):
            return len(self.available_idxs)
        elif(self.train_style == 'rollout'):
            return len(self.available_idxs)

    def pretrain(self):
        self._pretrain = True

    def pretrain_off(self):
        self._pretrain = False
    
    def __getitem__(self, idx):
        '''
        idx samples the file.
        Need to figure out a way to sample the snapshots within the file...
        '''
        if(self._pretrain and False):
            ###
            # Use this for pretraining
            ###
            if(self.train_style == 'next_step'):
                sim_idx = self.available_idxs[idx]      # Get valid prestored index
                sim_num = sim_idx // self.data.shape[1] # Get simulation number
                sim_time = sim_idx % self.data.shape[1] # Get time from that simulation

                # TODO: Compare multiple snapshots from same trajectory
                #       vs snapshots at same time from different trajectories
                #if(self.pretrain_type == "simulation"):
                if(False):
                    r_sim_idx = np.random.choice(self.available_idxs, 1)
                    #r_sim_idx = np.array([self.available_idxs[r] for r in rand_idxs])
                    r_sim_nums = list(r_sim_idx//self.data.shape[1])
                    #print(self.data.shape)
                    #print(r_sim_nums)
                    while(sim_num in r_sim_nums):
                        #rand_idxs = np.random.choice(self.available_idxs, 1)
                        #r_sim_idx = np.array([self.available_idxs[r] for r in rand_idxs])
                        r_sim_idx = np.random.choice(self.available_idxs, 1)
                        r_sim_nums = list(r_sim_idx//self.data.shape[1])
                    r_sim_nums.append(sim_num)
                    r_sim_nums = np.array(r_sim_nums)

                    targets = np.array([0,1])
                    shuffle_idxs = np.array([0,1])
                    np.random.shuffle(shuffle_idxs)

                    targets = targets[shuffle_idxs]
                    r_sim_nums = r_sim_nums[shuffle_idxs]

                    #print(r_sim_nums)
                    #print(self.data.shape)
                    #print(self.data[np.array(r_sim_nums)].shape)
                    #print(self.data[np.array(r_sim_nums)][:,sim_time][...,np.newaxis].shape)

                    #raise
                    return self.data[sim_num][sim_time-self.initial_step:sim_time], \
                            self.data[np.array(r_sim_nums)][:,sim_time][...,np.newaxis], \
                            self.grid[sim_num], \
                            self.all_tokens[idx].to(device=device), \
                            self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1], \
                            targets
                else:
                    # This will cause wraparound to next trajectory... probably what's causing issues
                    # Wraparound to next trajectory throws index error...
                    r_sim_times = list(np.random.choice([1,2,3,4,5], 2)) 
                    #r_sim_times = list(np.random.choice([1,2,3,4,5], 1)) 
                    r_sim_times.append(0)

                    # Wrap out of bounds back around to initial steps
                    r_sim_times = (np.array(r_sim_times) + sim_time)%self.data.shape[1]
                    targets = np.array([0,0,1])
                    shuffle_idxs = np.array([0,1,2])
                    #targets = np.array([0,1])
                    #shuffle_idxs = np.array([0,1])
                    #if(np.random.random() < 0.66):
                    if(np.random.random() < 0.82):
                        np.random.shuffle(shuffle_idxs)

                    targets = targets[shuffle_idxs]
                    r_sim_times = r_sim_times[shuffle_idxs] 
                    #print(shuffle_idxs)
                    #print(r_sim_times)
                    #print(targets)
                    #raise

                    return self.data[sim_num][sim_time-self.initial_step:sim_time], \
                            self.data[sim_num][np.array(r_sim_times)][...,np.newaxis], \
                            self.grid[sim_num], \
                            self.all_tokens[idx].to(device=device), \
                            self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1], \
                            targets
                target_frames = 0

                # Need to return target that indicates correct frame
                # Need to return multiple target frames
                if(self.return_text):
                            #self.data[sim_num][sim_time][...,np.newaxis], \
                    return self.data[sim_num][sim_time-self.initial_step:sim_time], \
                            self.data[sim_num][np.array(r_sim_times)][...,np.newaxis], \
                            self.grid[sim_num], \
                            self.all_tokens[idx].to(device=device), \
                            self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1], \
                            targets
                else:
                    if(sim_time == 0):
                        raise ValueError("WHOOPSIE")
                    return self.data[sim_num][sim_time - self.initial_step:sim_time], \
                                   self.data[sim_num][sim_time][np.newaxis], \

        # Everything is precomputed
        if(self.train_style == 'fixed_future'):
            #if(self._pretrain_tokens):
            #    return self.data[idx][:self.initial_step], \
            #       self.data[idx][self.sim_time][...,np.newaxis], \
            #       self.grid[idx], \
            #       self.all_tokens[idx].to(device=device), \
            #       self.time[idx][self.sim_time], \
            #       self.all_operator_maps[idx]
            #else:
            if(self.return_text):
                if(self._pretrain):
                    return self.data[idx][:self.initial_step], \
                       self.data[idx][self.sim_time][...,np.newaxis], \
                       self.grid[idx], \
                       self.all_tokens[idx].to(device=device), \
                       self.time[idx][self.sim_time], \
                       self.all_operator_maps[idx]
                else:
                    return self.data[idx][:self.initial_step], \
                       self.data[idx][self.sim_time][...,np.newaxis], \
                       self.grid[idx], \
                       self.all_tokens[idx].to(device=device), \
                       self.time[idx][self.sim_time], \
                       self.all_operator_maps[idx]
            else:
                #print(self.data[idx][:self.initial_step].shape)
                #print(self.data[idx][self.sim_time][...,np.newaxis].shape)
                #print(self.grid[idx].shape)
                return self.data[idx][:self.initial_step], \
                   self.data[idx][self.sim_time], \
                   self.grid[idx]
                   #self.data[idx][self.sim_time][...,np.newaxis], \
            #if(self.return_text):
            #    return self.data[idx][:self.initial_step], \
            #           self.data[idx][self.sim_time][...,np.newaxis], \
            #           self.grid[idx], \
            #           self.all_tokens[idx].to(device=device), \
            #           self.time[idx][self.sim_time]
            #else:
            #    return self.data[idx][...,:self.initial_step,:], \
            #           self.data[idx][self.sim_time], \
            #           self.grid[udx][self.sim_time]

        # Need to slice according to available data
        elif(self.train_style == 'next_step'):
            sim_idx = self.available_idxs[idx]      # Get valid prestored index
            sim_num = sim_idx // self.data.shape[1] # Get simulation number
            sim_time = sim_idx % self.data.shape[1] # Get time from that simulation

            if(self.return_text):
                return self.data[sim_num][sim_time-self.initial_step:sim_time], \
                   self.data[sim_num][sim_time][...,np.newaxis], \
                   self.grid[sim_num], \
                   self.all_tokens[sim_num].to(device=device), \
                   self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1], \
                   self.all_operator_maps[sim_num]
            else:
                #print(idx)
                #print(sim_idx)
                #print(sim_num, sim_time)
                #print(len(self.available_idxs))
                #print(self.data.shape)
                if(sim_time == 0):
                    raise ValueError("WHOOPSIE")
                return self.data[sim_num][sim_time-self.initial_step:sim_time], \
                   self.data[sim_num][sim_time], \
                   self.grid[sim_num]

        elif(self.train_style == 'arbitrary_step'):
            sim_idx = self.available_idxs[idx]      # Get valid prestored index
            #sim_idx = idx      # Get valid prestored index
            sim_num = sim_idx // self.data.shape[1] # Get simulation number
            sim_time = sim_idx % self.data.shape[1] # Get time from that simulation

            if(self.return_text):
                return self.data[sim_num][:self.initial_step], \
                   self.data[sim_num][sim_time][...,np.newaxis], \
                   self.grid[sim_num], \
                   self.all_tokens[sim_num].to(device=device), \
                   self.time[sim_num][sim_time], \
                   self.all_operator_maps[sim_num]
                #return self.data[sim_num][:self.initial_step], \
                #   self.data[sim_num][sim_time][...,np.newaxis], \
                #   self.grid[sim_num], \
                #   self.all_tokens[sim_num].to(device=device), \
                #   self.time[sim_num][sim_time], \
                #return self.data[sim_num][sim_time-self.initial_step:sim_time], \
            else:
                return self.data[sim_num][:self.initial_step], \
                   self.data[sim_num][sim_time][...,np.newaxis], \
                   self.grid[sim_num], \
                   self.all_tokens[sim_num].to(device=device), \
                   self.time[sim_num][sim_time], \
                   self.all_operator_maps[sim_num]
                #return self.data[sim_num][sim_time-self.initial_step:sim_time,...][...,np.newaxis], \
                #       self.data[sim_num][sim_time][...,np.newaxis], \
                #       self.grid[sim_num][...,np.newaxis]

        # Need to slice according ot available data and rollout
        elif(self.train_style == 'rollout'):
            sim_idx = self.available_idxs[idx]      # Get valid prestored index
            sim_num = sim_idx // self.data.shape[1] # Get simulation number
            sim_time = sim_idx % self.data.shape[1] # Get time from that simulation
            if(self.return_text):
                # Add additional times to text encoding.
                slice_times = self.time[sim_num][sim_time-self.initial_step:sim_time+self.rollout_length] # Get times
                #print(sim_time, sim_time - self.initial_step, sim_time + self.rollout_length, self.initial_step, self.rollout_length)
                slice_tokens = torch.empty((len(slice_times), 15))
                for idx, st in enumerate(slice_times):
                    # Loses a very small amount of precision
                    # Need predefined tensor
                    slce = self._encode_tokens("&" + str(st))
                    if(len(slce) < 15):
                        slce.extend([20.]*(15-len(slce)))
                    slice_tokens[idx] = torch.Tensor(slce)[:15].to(device=device)#.cuda()

                # This goes into ssl training loop.
                return_tokens = self.tokens[sim_num].copy()
                return_tokens.extend([len(self.WORDS)]*(500 - len(return_tokens)))
                return_tokens = torch.Tensor(return_tokens)
                return_tokens = return_tokens.repeat(self.rollout_length, 1)
                slice_tokens = torch.swapaxes(slice_tokens.unfold(0, 10, 1)[:-1], 1, 2).reshape(self.rollout_length, -1)
                all_tokens = torch.cat((return_tokens, slice_tokens), dim=1)

                # Most processing happens in the training loop
                return self.data[sim_num][sim_time-self.initial_step:sim_time+self.rollout_length,...][...,np.newaxis], \
                       self.data[sim_num][sim_time:sim_time+self.rollout_length][...,np.newaxis], \
                       self.grid[sim_num][...,np.newaxis], \
                       all_tokens
                       #return_tokens, slice_tokens
            else:
                return self.data[sim_num][sim_time-self.initial_step:sim_time,...][...,np.newaxis], \
                       self.data[sim_num][sim_time:sim_time+self.rollout_length], \
                       self.grid[sim_num][...,np.newaxis]


class TransformerMultiOperatorDataset(Dataset):
    def __init__(self, base_path,
                 initial_step=10,
                 saved_folder='./data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 num_t=200,
                 num_x=200,
                 sim_time=-1,
                 split="train",
                 test_ratio=0.2,
                 val_ratio=0.2,
                 num_samples=None,
                 return_text=False,
                 rollout_length=10,
                 train_style='fixed_future',
                 seed=None,
                 augment=False, ssl=False, forcing_term='full', flnm='all',
                 ):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """

        # Define path to files
        #self.file_path = os.path.abspath(f.filename)
        #self.file_path = os.path.abspath(saved_folder + filename + ".h5")
        self.return_text = return_text
        self.train_style = train_style
        self.augment = augment
        self.ssl = ssl
        self.forcing = False
        self._pretrain_tokens = False
        self.reduced_resolution = reduced_resolution
        self.reduced_resolution_t = reduced_resolution_t


        #data_files = ['varied_heat_10000.h5', 'varied_burgers_2500.h5', 'varied_kdv_2500.h5']

        #data_files = ['kdv_1000.h5', 'heat_1000.h5', 'burgers_1000.h5']
        #data_files = ['heat_1000.h5', 'burgers_1000.h5']
        #data_files = ['heat_250.h5', 'burgers_250.h5', 'kdv_250.h5']
        #data_files = ['heat_250.h5', 'burgers_250.h5', 'new_kdv_250.h5']
        self.forcing_term = forcing_term
        #print("\nFORCING TERM: {}\n".format(self.forcing_term))
        #raise
        if(self.forcing_term == 'full'):
            data_files = ['heat_2000.h5', 'burgers_250.h5', 'new_kdv_250.h5']
        elif(self.forcing_term == 'non_td'):
            #data_files = ['non_td_heat_1000.h5', 'non_td_burgers_250.h5', 'non_td_new_kdv_250.h5']
            data_files = ['xwide_non_td_heat_2000.h5', 'xwide_non_td_burgers_250.h5', 'xwide_non_td_kdv_250.h5']
        elif(self.forcing_term == 'none'):
            #data_files = ['non_td_heat_1000.h5', 'non_td_burgers_250.h5', 'non_td_new_kdv_250.h5']
            if(flnm == 'all'):
                data_files = ['new_long_xwide_no_forcing_heat_2000.h5',
                              'new_long_xwide_no_forcing_burgers_250.h5',
                              'new_long_xwide_no_forcing_advection_2000.h5',
                              ]#,
                              #'new_long_xwide_no_forcing_kdv_500.h5']#, 'xwide_no_forcing_ks_250.h5']
                print("\n\nNO FORCING ALL DATA\n\n")
            elif(flnm == 'Burgers'):
                data_files = ['non_td_burgers_250.h5']
            elif(flnm == 'Heat'):
                data_files = ['non_td_heat_1000.h5']
            else:
                raise ValueError("Invalid dataset choice.")

        elif(self.forcing_term == 'all'):
            #data_files = ['non_td_heat_1000.h5', 'non_td_burgers_250.h5', 'non_td_new_kdv_250.h5']
            if(flnm == 'all'):
                data_files = ['new_long_xwide_no_forcing_heat_2000.h5',
                              'new_long_xwide_no_forcing_burgers_250.h5',
                              'new_long_xwide_no_forcing_advection_2000.h5',
                              'varied_heat_10000.h5', 'varied_burgers_2500.h5', 'varied_kdv_2500.h5'
                              ]#,
                              #'new_long_xwide_no_forcing_kdv_500.h5']#, 'xwide_no_forcing_ks_250.h5']
                print("\n\nNO FORCING ALL DATA\n\n")
            elif(flnm == 'Burgers'):
                data_files = ['non_td_burgers_250.h5']
            elif(flnm == 'Heat'):
                data_files = ['non_td_heat_1000.h5']
            else:
                raise ValueError("Invalid dataset choice.")

            #data_files = ['non_td_burgers_250.h5']
            #data_files = ['non_td_burgers_250.h5']
        else:
            raise ValueError("Invalid forcing term selection. Select 'full, 'non_td', or 'none'.")

        if(self.train_style == 'next_step' and sim_time == -1):
            self.sim_time = num_t//self.reduced_resolution_t - 1
        elif(self.train_style == 'arbitrary_step' and sim_time == -1):
            self.sim_time = num_t//self.reduced_resolution_t - 1
        else:
            self.sim_time = sim_time

        self.data = []
        self.grid = []
        self.time = []
        self.tokens = []
        self.available_idxs = []
        self.all_data_list = []
        self.all_operator_maps = []
        for df in data_files:
            print("\nDATA FILE: {}".format(df))
            f = h5py.File("{}{}".format(base_path, df), 'r')

            # Get data list
            torch.manual_seed(seed)
            np.random.seed(seed)
            if('heat' in df):
                print(list(f.keys())[0])
                if(len(list(f.keys())[0].split("_")) == 3):
                    data_list = [key for key in f.keys()]
                elif(len(list(f.keys())[0].split("_")) == 4):
                    #data_list = ['_'.join(k for k in np.array(key.split("_"))[np.array([0,1,3])]) for key in f.keys()]
                    #print(data_list)
                    data_list = [key for key in f.keys()]
                #data_list = [key for key in f.keys() if(len(key.split("_")) == 3) else '_'.join(k for k in key.split("_")[0,1,4])]
                #print(data_list)
                #raise
            else:
                data_list = [key for key in f.keys()]
            np.random.shuffle(data_list)

            self.data_list = data_list

            # Get target split. Seeding is required to make this reproducible.
            # This splits each run, lets try a better shuffle
            if(num_samples is not None):
                data_list = data_list[:num_samples]
            train_idx = int(len(data_list) * (1 - test_ratio - val_ratio))
            val_idx = int(len(data_list) * (1-test_ratio))
            if(split == "train"):
                self.data_list = np.array(data_list[:train_idx])
            elif(split == "val"):
                self.data_list = np.array(data_list[train_idx:val_idx])
            elif(split == "test"):
                self.data_list = np.array(data_list[val_idx:])
            else:
                raise ValueError("Select train, val, or test split. {} is invalid.".format(split))

            # Hold on to all of the datas
            self.all_data_list.append(self.data_list)

            ###
            # Create multi-class label for each equation.
            #
            # [1,0,0] corresponds to nonlinear advection term (Burgers, KdV)
            # [0,1,0] corresponds to diffusion term (Burgers, Heat)
            # [0,0,1] corresponds to third order term (KdV)
            #
            ###
            #o_map = [[1,1,0] if('burgers' in df) else [0,1,0] if('heat' in df) else [1,0,1]] * len(self.data_list)
            #self.all_operator_maps.append(o_map)

            # Time steps used as initial conditions
            self.initial_step = initial_step
            self.rollout_length = rollout_length

            self.WORDS = ['(', ')', '+', '-', '*', '/', 'Derivative', 'Sum', 'j', 'A_j', 'l_j',
                     'omega_j', 'phi_j'    , 'sin', 't', 'u', 'x', 'dirichlet', 'neumann',
                     "None", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10^',
                     #'E', ',', '.', '&']
                     'E', 'e', ',', '.', '&']
            self.word2id = {w: i for i, w in enumerate(self.WORDS)}
            self.id2word = {i: w for i, w in enumerate(self.WORDS)}
            self.num_x = num_x
            self.num_t = num_t
                       
            if('kdv' in df):
                self.num_x = 400 
            if('varied' in df):
                self.num_x = 100
                self.num_t = 100
                self.name = "pde_{}-{}".format(self.num_t, self.num_x)
                self.reduced_resolution_t = 2
                self.reduced_resolution = 2
            else:
                self.name = "pde_{}-{}".format(self.num_t, self.num_x)
                self.reduced_resolution_t = 4
                self.reduced_resolution = 4
            #self.name = "pde_{}-{}".format(self.num_x, self.num_t)

            #self.h5_file = h5py.File(self.file_path, 'r')
            #self.sim_time = sim_time

            #print(len(self.data_list))
            #raise
            for i in tqdm(range(len(self.data_list))):
                seed_group = f[self.data_list[i]]
                if('kdv' in df):
                    self.data.append(seed_group[self.name][0][:,::2][::self.reduced_resolution_t,::self.reduced_resolution])
                else:
                    self.data.append(seed_group[self.name][0][::self.reduced_resolution_t,::self.reduced_resolution])

                if(self.train_style == 'next_step'):
                    idxs = np.arange(0, len(seed_group[self.name][0])//self.reduced_resolution_t)[self.initial_step:self.sim_time]
                elif(self.train_style == 'arbitrary_step'):
                    #idxs = np.arange(0, len(seed_group[self.name][0]))[self.initial_step:]
                    idxs = np.arange(0, len(seed_group[self.name][0])//self.reduced_resolution_t)[self.initial_step:self.sim_time]

                #else:#(self.train_style == 'rollout'):
                #    length = len(seed_group[self.name][0])
                #    idxs = np.arange(0, length)[self.initial_step:length-self.rollout_length]
                elif(self.train_style == 'rollout'):
                    length = len(seed_group[self.name][0])
                    idxs = np.arange(0, length)[self.initial_step:length-self.rollout_length]
                elif(self.train_style == 'fixed_future'):
                    idxs = np.array([i])
                else:
                    raise

                if(len(self.available_idxs) != 0):
                    idxs += self.num_t//self.reduced_resolution_t - (self.sim_time-1) + self.available_idxs[-1] if(self.train_style in ['next_step', 'arbitrary_step']) else \
                            self.available_idxs[-1] + 1 + self.rollout_length if(self.train_style == 'rollout') else \
                    self.available_idxs[-1] + len(seed_group[self.name][0]) - self.sim_time + 1
                self.available_idxs.extend(idxs)
                #print(idxs)

                if('kdv' in df):
                    self.grid.append(np.array(seed_group[self.name].attrs["x"], dtype='f')[::2][::self.reduced_resolution])
                else:
                    self.grid.append(np.array(seed_group[self.name].attrs["x"], dtype='f')[::self.reduced_resolution])

                # Get operator coefficients
                if(self.return_text):
                    self.tokens.append(torch.Tensor(seed_group[self.name].attrs['encoded_tokens']))
                    self.time.append(seed_group[self.name].attrs['t'][::self.reduced_resolution_t])
                    dl_split = self.data_list[i].split("_")
                    if(dl_split[0] == 'Burgers'):
                        omap = [float(dl_split[2]), float(dl_split[3]), 0]
                    elif(dl_split[0] == 'Heat'):
                        omap = [0, float(dl_split[2]), 0]
                    elif(dl_split[0] == 'KdV'):
                        omap = [float(dl_split[2]), 0, float(dl_split[3])]
                    elif(dl_split[0] == 'Advection'):
                        omap = [0, 0, float(dl_split[2])]
                    else:
                        raise ValueError("Invalid 1D data set used. Only Heat, Burgers, and KdV are currently supported.")
                    self.all_operator_maps.append(omap)

            f.close()
            #raise

        #raise
        #print(self.available_idxs)
        self.data = torch.Tensor(np.array(self.data))#.to(device=device)
        #raise
        self.grid = torch.Tensor(np.array(self.grid))#.to(device=device)
        self.all_data_list = np.array(self.all_data_list).flatten()
        self.all_operator_maps = torch.Tensor(np.array(self.all_operator_maps).reshape((-1,3))).float()
        #print(self.available_idxs)
        #print(self.all_data_list)
        #raise

        print("\nNUMBER OF SAMPLES: {}".format(len(self.available_idxs)))
        #raise

        def forcing_term(x, t, As, ls, phis, omegas):
            return np.sum(As[i]*torch.sin(2*np.pi/16. * ls[i]*x + omegas[i]*t + phis[i]) for i in range(len(As)))

        # Not suitable for autoregressive training
        if(self.train_style == 'fixed_future'):
            self.all_tokens = torch.empty(len(self.data), 500)#.to(device=device)#.cuda()
            for idx, token in tqdm(enumerate(self.tokens)):
                if(self.return_text):
                    # Encode time token
                    slice_tokens = self._encode_tokens("&" + str(self.time[idx][self.sim_time]))
                    return_tokens = torch.Tensor(self.tokens[idx].clone()).cuda()
                    return_tokens = torch.cat((return_tokens, torch.Tensor(slice_tokens).cuda()))
                    return_tokens = torch.cat((return_tokens, torch.Tensor([len(self.WORDS)]*(500 - len(return_tokens))).cuda()))
                    self.all_tokens[idx] = return_tokens.to(device=device)#.cuda()


        elif(self.train_style in ['next_step', 'arbitrary_step'] and self.return_text):
            # Create array of all legal encodings, pdes, and data
            self.all_tokens = torch.empty(len(self.available_idxs), 500)#.to(device=device)#.cuda()

            #print()
            #print()
            #print(self.available_idxs)
            #print()
            #print()
            #raise
            #print(len(self.time))
            #print(self.time[0].shape)
            #print(self.data.shape)
            #raise
            for idx, sim_idx in tqdm(enumerate(self.available_idxs)):
                #sim_idx = self.available_idxs[idx]      # Get valid prestored index
                sim_num = sim_idx // self.data.shape[1] # Get simulation number
                sim_time = sim_idx % self.data.shape[1] # Get time from that simulation
                if(self.return_text):
                    try:
                        slice_tokens = self._encode_tokens("&" + str(self.time[sim_num][sim_time]))
                    except IndexError:
                        print()
                        print()
                        print(self.data.shape, len(self.time), len(self.time[0]), sim_num, sim_time)
                        print()
                        print()
                        raise
                    return_tokens = torch.Tensor(self.tokens[sim_num].clone())

                    # TODO: Maybe put this back
                    return_tokens = torch.cat((return_tokens, torch.Tensor(slice_tokens))).cuda()

                    return_tokens = torch.cat((return_tokens, torch.Tensor([len(self.WORDS)]*(500 - len(return_tokens))).cuda()))
                    self.all_tokens[idx] = return_tokens.to(device=device)#.cuda()

        if(self.return_text):
            self.all_tokens = self.all_tokens#.to(device=device)#.cuda()
        self.time = torch.Tensor(self.time)#.to(device=device)
        if(self.augment):
            print("\nNumber of augmented samples: {}\n".format(len(self.all_tokens)))
        print()
        print()
        print(self.data.shape)
        print()
        print()

    def _encode_tokens(self, all_tokens):
        encoded_tokens = []
        num_concat = 0
        for i in range(len(all_tokens)):
            try: # All the operators, bcs, regular symbols
                encoded_tokens.append(self.word2id[all_tokens[i]])
                if(all_tokens[i] == "&"): # 5 concatenations before we get to lists of sampled values
                    num_concat += 1
            except KeyError: # Numerical values
                if(isinstance(all_tokens[i], str)):
                    for v in all_tokens[i]:
                        try:
                            encoded_tokens.append(self.word2id[v])
                        except KeyError:
                            print(all_tokens)
                            raise
                    if(num_concat >= 5): # We're in a list of sampled parameters
                        encoded_tokens.append(self.word2id[","])
                else:
                    raise KeyError("Unrecognized token: {}".format(all_tokens[i]))

        return encoded_tokens

    def _one_hot_encode(self, tokens):
        encoding = np.zeros((len(tokens), len(self.WORDS)+1))
        encoding[range(tokens.shape[0]), tokens] = 1
        return encoding

    def __len__(self):
        if(self.train_style == 'fixed_future'):
            return self.data.shape[0]
        elif(self.train_style in ['next_step', 'arbitrary_step']):
            #if(len(self.all_tokens.shape) == 3):
            if(self.augment and self.ssl):
                return len(self.available_idxs)
            elif(self.augment):
                return len(self.all_tokens_map)
            else:
                return len(self.available_idxs)
        elif(self.train_style == 'rollout'):
            return len(self.available_idxs)

    def pretrain(self):
        self._pretrain_tokens = True

    def pretrain_off(self):
        self._pretrain_tokens = False

    def __getitem__(self, idx):
        '''
        idx samples the file.
        Need to figure out a way to sample the snapshots within the file...
        '''
        #print("\n\nHERE\n\n")
        #print("\nHERE\n")
        # Everything is precomputed
        if(self.train_style == 'fixed_future'):
            if(self.return_text):
                if(self._pretrain_tokens):
                    return self.data[idx][:self.initial_step], \
                       self.data[idx][self.sim_time][...,np.newaxis], \
                       self.grid[idx], \
                       self.all_tokens[idx].to(device=device), \
                       self.time[idx][self.sim_time], \
                       self.all_operator_maps[idx]
                else:
                    return self.data[idx][:self.initial_step], \
                       self.data[idx][self.sim_time][...,np.newaxis], \
                       self.grid[idx], \
                       self.all_tokens[idx].to(device=device), \
                       self.time[idx][self.sim_time]#, \
            else:
                return self.data[idx][:self.initial_step], \
                       self.data[idx][self.sim_time], \
                       self.grid[idx]

        # Need to slice according to available data
        elif(self.train_style == 'next_step'):
            sim_idx = self.available_idxs[idx]      # Get valid prestored index
            #sim_idx = idx      # Get valid prestored index
            sim_num = sim_idx // self.data.shape[1] # Get simulation number
            sim_time = sim_idx % self.data.shape[1] # Get time from that simulation

            if(self.return_text):
                if(self._pretrain_tokens):
                    #print(self.data.shape)
                    #print(self.grid.shape)
                    #print("\n\nHERE??\n\n")
                    #print(self.time[sim_num][self.sim_time] - self.sim_num][sim_time-1]
                    return self.data[sim_num][sim_time-self.initial_step:sim_time], \
                       self.data[sim_num][sim_time][...,np.newaxis], \
                       self.grid[sim_num], \
                       self.all_tokens[sim_num].to(device=device), \
                       self.time[sim_num][sim_time], \
                       self.all_operator_maps[sim_num]
                       #self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1], \
                else:
                    return self.data[sim_num][sim_time-self.initial_step:sim_time], \
                       self.data[sim_num][sim_time][...,np.newaxis], \
                       self.grid[sim_num], \
                       self.all_tokens[sim_num].to(device=device), \
                       self.time[sim_num][sim_time], \
                       self.all_operator_maps[sim_num]
                       #self.time[sim_num][self.sim_time] - self.time[sim_num][sim_time-1]
            else:
                if(sim_time == 0):
                    raise ValueError("WHOOPSIE")
                return self.data[sim_num][sim_time-self.initial_step:sim_time], \
                               self.data[sim_num][sim_time][...,np.newaxis], \
                               self.grid[sim_num]
                           #self.all_tokens[idx].to(device=device), \
                           #self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1]#, \
                           #self.data[sim_num][sim_time-self.initial_step:sim_time,...][...,np.newaxis]
                #return self.data[sim_num][sim_time-self.initial_step:sim_time,...][...,np.newaxis], \
                #       self.data[sim_num][sim_time][...,np.newaxis], \
                #       self.grid[sim_num][...,np.newaxis]

        elif(self.train_style == 'arbitrary_step'):
            sim_idx = self.available_idxs[idx]      # Get valid prestored index
            #sim_idx = idx      # Get valid prestored index
            sim_num = sim_idx // self.data.shape[1] # Get simulation number
            sim_time = sim_idx % self.data.shape[1] # Get time from that simulation

            if(self.return_text):
                #print(sim_idx, sim_num, sim_time)
                return self.data[sim_num][:self.initial_step], \
                   self.data[sim_num][sim_time][...,np.newaxis], \
                   self.grid[sim_num], \
                   self.all_tokens[sim_num].to(device=device), \
                   self.time[sim_num][sim_time], \
                   self.all_operator_maps[sim_num]
                #return self.data[sim_num][sim_time-self.initial_step:sim_time], \
                #return self.data[sim_num][0], \
                #   self.data[sim_num][sim_time][...,np.newaxis], \
                #   self.grid[sim_num], \
                #   self.all_tokens[idx].to(device=device), \
                #   self.time[sim_num][sim_time]# - self.time[sim_num][sim_time-1]#, \
                   #self.data[sim_num][sim_time-self.initial_step:sim_time,...][...,np.newaxis]
            else:
                return self.data[sim_num][sim_time-self.initial_step:sim_time,...][...,np.newaxis], \
                       self.data[sim_num][sim_time][...,np.newaxis], \
                       self.grid[sim_num][...,np.newaxis]

        # Need to slice according ot available data and rollout
        elif(self.train_style == 'rollout'):
            sim_idx = self.available_idxs[idx]      # Get valid prestored index
            sim_num = sim_idx // self.data.shape[1] # Get simulation number
            sim_time = sim_idx % self.data.shape[1] # Get time from that simulation
            if(self.return_text):
                # Add additional times to text encoding.
                slice_times = self.time[sim_num][sim_time-self.initial_step:sim_time+self.rollout_length] # Get times
                #print(sim_time, sim_time - self.initial_step, sim_time + self.rollout_length, self.initial_step, self.rollout_length)
                slice_tokens = torch.empty((len(slice_times), 15))
                for idx, st in enumerate(slice_times):
                    # Loses a very small amount of precision
                    # Need predefined tensor
                    slce = self._encode_tokens("&" + str(st))
                    if(len(slce) < 15):
                        slce.extend([20.]*(15-len(slce)))
                    slice_tokens[idx] = torch.Tensor(slce)[:15].to(device=device)#.cuda()

                # This goes into ssl training loop.
                return_tokens = self.tokens[sim_num].copy()
                return_tokens.extend([len(self.WORDS)]*(500 - len(return_tokens)))
                return_tokens = torch.Tensor(return_tokens)
                return_tokens = return_tokens.repeat(self.rollout_length, 1)
                slice_tokens = torch.swapaxes(slice_tokens.unfold(0, 10, 1)[:-1], 1, 2).reshape(self.rollout_length, -1)
                all_tokens = torch.cat((return_tokens, slice_tokens), dim=1)

                # Most processing happens in the training loop
                return self.data[sim_num][sim_time-self.initial_step:sim_time+self.rollout_length,...][...,np.newaxis], \
                       self.data[sim_num][sim_time:sim_time+self.rollout_length][...,np.newaxis], \
                       self.grid[sim_num][...,np.newaxis], \
                       all_tokens
                       #return_tokens, slice_tokens
            else:
                return self.data[sim_num][sim_time-self.initial_step:sim_time,...][...,np.newaxis], \
                       self.data[sim_num][sim_time:sim_time+self.rollout_length], \
                       self.grid[sim_num][...,np.newaxis]


class TransformerOperatorDataset2D(Dataset):
    def __init__(self, f, 
                 initial_step=10,
                 saved_folder='./data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 num_t=200,
                 num_x=200,
                 sim_time=-1,
                 split="train",
                 test_ratio=0.2,
                 val_ratio=0.2,
                 num_samples=None,
                 return_text=False,
                 train_style='fixed_future',
                 rollout_length=10,
                 split_style='equation',
                 samples_per_equation=111,
                 seed=0,
                 pretrain=False,
                 pad_length=100
                 ):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """
        
        # Define path to files
        self.file_path = os.path.abspath(f.filename)
        self.return_text = return_text
        self.train_style = train_style
        self.rollout_length = rollout_length
        self.split_style = split_style
        self.samples_per_equation = samples_per_equation
        self._pretrain = pretrain
        
        # Extract list of seeds
        self.data_list = list(f.keys())

        # Time steps used as initial conditions
        self.initial_step = initial_step

        self.WORDS = ['(', ')', '+', '-', '*', '/', '=', 'Derivative', 'sin', 'cos', 't', 'u', 'x', 'w', 'y',
                      'pi', 'Delta', 'nabla', 'dot', "None", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10^',
                      'E', 'e', ',', '.', '&']
        self.word2id = {w: i for i, w in enumerate(self.WORDS)}
        self.id2word = {i: w for i, w in enumerate(self.WORDS)}

        self.num_t = num_t
        self.num_x = num_x
        #self.name = "pde_{}-{}".format(self.num_t, self.num_x)

        self.h5_file = h5py.File(self.file_path, 'r')
        self.sim_time = sim_time

        sample_num = 0
        # Get all indices
        idxs = []
        #TODO this shuffles by EQUATION, need to shuffle by SIMULATION?
        for i in range(len(self.data_list)):
            seed_group = self.h5_file[self.data_list[i]]
            samples_per_sim = seed_group['u'].shape[0]
            for j in range(seed_group['u'].shape[0]):
                idxs.append(i*seed_group['u'].shape[0] + j)
        #print(self.data_list)
        idxs = [i for i in range(len(self.data_list))] #TODO 

        print("\nSEED: {}".format(seed))
        np.random.seed(seed)
        np.random.shuffle(idxs)
        self.idxs = idxs[:num_samples]

        # Split indices into
        #print(idxs)
        #raise
        if(self.split_style == 'equation'):
            train_idx = int(num_samples * (1 - test_ratio - val_ratio))
            val_idx = int(num_samples * (1-test_ratio))
            if(split == "train"):
                self.idxs = self.idxs[:train_idx]
            elif(split == "val"):
                self.idxs = self.idxs[train_idx:val_idx]
            elif(split == "test"):
                self.idxs = self.idxs[val_idx:num_samples]
            else:
                raise ValueError("Select train, val, or test split. {} is invalid.".format(split))
        #print(self.idx)


        self.data = []
        self.grid = []
        self.time = []
        self.w0 = []
        self.temp_tokens = []
        self.available_idxs = []
        self.data_list = np.array(self.data_list)[self.idxs]
        self.o_map = []
        #self.data_list = np.array([self.data_list[0]])
        #print(self.data_list)
        #self.data = torch.empty((40,200,64,64,201)).float()
        #self.data = torch.empty((len(self.data_list),200,64,64,201)).float()
        #print(vars(self.h5_file))
        #print(self.h5_file.filename)
        if('10s' in self.h5_file.filename):
            self.data = torch.empty((len(self.data_list),self.samples_per_equation,64,64,201)).float()
        elif('30s' in self.h5_file.filename):
            self.data = torch.empty((len(self.data_list),self.samples_per_equation,64,64,121)).float()
        elif('1s' in self.h5_file.filename):
            #print(len(self.data_list))
            self.data = torch.empty((len(self.data_list),self.samples_per_equation,64,64,201)).float()
            #raise
        for i in tqdm(range(len(self.data_list))):
            seed_group = self.h5_file[self.data_list[i]]
            #print(np.log(float(self.data_list[i].split("_")[0])))
            #self.o_map.append(torch.Tensor([1., np.log(float(self.data_list[i].split("_")[0]))]))
            #self.o_map.append(torch.Tensor([1., np.log(float(self.data_list[i].split("_")[0]))]))
            #self.o_map.append(torch.Tensor([np.log(float(self.data_list[i].split("_")[0])), 1., 1., 1.,]))
            print(self.data_list[i])
            sdl = self.data_list[i].split("_")
            DIFF_MAX = 10e-5
            ADV_MAX = 2
            if('Burgers' in self.data_list[i]):
                self.o_map.append(torch.Tensor([float(sdl[1])/DIFF_MAX, float(sdl[2])/ADV_MAX, float(sdl[3])/ADV_MAX]))
            elif('Heat' in self.data_list[i]):
                self.o_map.append(torch.Tensor([float(sdl[1])/DIFF_MAX, 0., 0.]))
            data = seed_group['u'][:][::reduced_resolution_t,::reduced_resolution,::reduced_resolution,...]

            print(self.o_map[-1])
            #print(data.shape)
            #raise

            # Get extra info
            base_tokens = seed_group['tokens'][:]
            x = seed_group['X'][:][::reduced_resolution_t,::reduced_resolution,::reduced_resolution,np.newaxis]
            y = seed_group['Y'][:][::reduced_resolution_t,::reduced_resolution,::reduced_resolution,np.newaxis]
            w0 = seed_group['a'][:][...,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,np.newaxis]

            # Add initial condition
            complete_data = np.concatenate((w0, data), axis=3)
            #self.data.append(torch.Tensor(complete_data).clone())
            #complete_data = complete_data[...,:101]
            #print(complete_data.shape)
            #raise
            self.data[i] = torch.Tensor(complete_data[:self.samples_per_equation])
            #print(complete_data.shape)

            # Add initial time
            time = list(seed_group['t'][:])
            time.insert(0, 0.0)
            self.time.append(time)

            # Get grid
            self.grid.append(np.dstack((x,y)))

            # Get tokens
            #print(base_tokens)

            ###
            # TODO: Double check this.
            ###
            # Correct for small issue.
            base_tokens[38] = 12
            #base_tokens = torch.cat((base_tokens[:33], torch.Tensor([16]), base_tokens[33:]))
            base_tokens = np.insert(base_tokens, 34, 16)
            #base_tokens[8] = 11
            #base_tokens[17] = 11
            #base_tokens[35] = 11
            #print(base_tokens)
            #raise
            #raise
            self.temp_tokens.append(base_tokens)
            #print("\nGOT SAMPLE {}\n".format(i))
            del complete_data

        # Arrange data
        print("ARRANGING DATA")
        self.data = torch.swapaxes(self.data, 2, 4)
        self.data = torch.swapaxes(self.data, 3, 4)
        print("\n\nDATA SHAPE:")
        print(self.data.shape)
        #print()
        #print()
        #raise

        # Get valid indices for returning data
        #print("Getting available idxs...")
        self.available_idxs = []
        #print(len(self.data_list))
        #raise
        if(self.train_style in ['next_step', 'arbitrary_step']):
            for i in tqdm(range(len(self.data_list))):
                if(self.train_style == 'next_step'):
                    idxs = np.arange(0, self.data.shape[2])[self.initial_step:]
                    if(self.split_style == 'equation'):
                        for j in range(1, self.samples_per_equation):
                            idxs = np.append(idxs, np.arange(0, self.data.shape[2])[self.initial_step:] + idxs[-1]+1)
                elif(self.train_style == 'arbitrary_step'):
                    idxs = np.arange(0, self.data.shape[2])[self.initial_step:]
                
                # Take into account that the first self.initial_step samples can't be used as target
                if(len(self.available_idxs) != 0): #TODO Make this robust to initial step
                    idxs += self.available_idxs[-1] + 1 if(self.train_style == 'next_step') else \
                            self.available_idxs[-1] + 1 + self.rollout_length if(self.train_style == 'rollout') else \
                                                    self.available_idxs[-1] + 1
                self.available_idxs.extend(idxs)

        elif(self.train_style == 'fixed_future'): # Only need to keep track of total number of valid samples
            idxs = np.arange(0, self.data.shape[0]*self.data.shape[1])
            self.available_idxs = idxs

        # Flatten data to combine simulations
        self.data = self.data.flatten(start_dim=0, end_dim=1)

        # Grid to tensor
        self.grid = torch.Tensor(np.array(self.grid))

        # Add tokenized time to each equation for each simulation
        #print("Getting tokens...")
        self.tokens = []
        self.tokens = torch.empty(len(self.time), self.data.shape[1], pad_length)
        for idx, token in enumerate(self.temp_tokens):
            for jdx, time in enumerate(self.time[idx]):

                # Tokenize time
                slice_tokens = self._encode_tokens("&" + str(time))

                # Add tokenized time to equation
                full_tokens = copy.copy(list(token))
                full_tokens.extend(list(slice_tokens))

                # Pad tokens to all have same length
                full_tokens.extend([len(self.WORDS)]*(pad_length - len(full_tokens)))

                # Hold on to tokens
                self.tokens[idx][jdx] = torch.Tensor(full_tokens)

        # Time and tokens to tensors
        self.time = torch.Tensor(np.array(self.time))
        self.tokens = torch.Tensor(self.tokens)

        if(self.split_style == 'initial_condition'):
            #train_idx = int(len(self.available_idxs) * (1 - test_ratio - val_ratio))
            train_idx = int(self.data.shape[0] * (1 - test_ratio - val_ratio))
            val_idx = int(self.data.shape[0] * (1-test_ratio))
            #self.idxs = [i for i in range(len(self.available_idxs))]
            self.idxs = [i for i in range(self.data.shape[0])]
            np.random.shuffle(self.idxs)
            #print(train_idx, val_idx, num_samples, len(self.idxs))
            if(split == "train"):
                self.idxs = self.idxs[:train_idx]
            elif(split == "val"):
                self.idxs = self.idxs[train_idx:val_idx]
            elif(split == "test"):
                self.idxs = self.idxs[val_idx:]
            else:
                raise ValueError("Select train, val, or test split. {} is invalid.".format(split))
            self.idx_to_avail_map = {i[0]: i[1] for i in zip(self.idxs, self.available_idxs)}
            self.sample_to_idx_map = {i[0]: i[1] for i in zip(self.idxs, self.available_idxs)}

        self.h5_file.close()
        print("DATA SHAPE: {}".format(self.data.shape))
        print("NUM AVAILABLE IDXS: {}".format(len(self.available_idxs)))
        print("NUM IDXS: {}".format(len(self.idxs)))
        print("{} good samples.".format(len(self.data)))
        print(self.split_style)
        print(self.train_style)

        # Create data tuples
        self.data_tuples = []
        dt = self.time[0][1] - self.time[0][0] # TODO Assumes single timestep
        if(self.split_style == 'initial_condition'):
            if(self.train_style == 'next_step'):
                #for idx in range(len(self.idxs)):
                for idx in self.idxs:
                    #print(self.idxs)
                    #print(self.data.shape)
                    #raise
                    for jdx in range(self.initial_step, self.data.shape[1]):
                    #for jdx in range(self.initial_step, 101):
                        #idx = self.idx_to_avail_map[self.idxs[idx]]

                        #print(self.data.shape)
                        sim_idx = self.available_idxs[idx]
                        sim_num = sim_idx // self.data.shape[1] # Get simulation number
                        sim_time = sim_idx % self.data.shape[1] # Get time from that simulation

                        self.data_tuples.append((self.data[idx][jdx-self.initial_step:jdx],
                                self.data[idx][jdx][...,np.newaxis],
                                self.grid[idx//self.samples_per_equation],
                                self.tokens[idx//self.samples_per_equation][jdx], dt, self.o_map[idx//self.samples_per_equation]))
                                #self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1]))
                        #self.data_tuples.append((self.data[sim_num][sim_time-self.initial_step:sim_time],
                        #        self.data[sim_num][sim_time][...,np.newaxis],
                        #        self.grid[sim_num],
                        #        self.tokens[sim_num][sim_time],
                        #        self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1]))
            elif(self.train_style == 'fixed_future'):
                #for idx in tqdm(range(self.data.shape[0])):
                for idx in tqdm(self.idxs):
                    sim_num = idx // self.data.shape[1] # Get simulation number
                    sim_time = idx % self.data.shape[1] # Get time from
                    #print(idx, sim_num, sim_time, self.data.shape)
                    self.data_tuples.append((
                        self.data[idx][:self.initial_step],
                        self.data[idx][self.sim_time].unsqueeze(-1),
                        self.grid[sim_num],
                        self.tokens[sim_num][self.sim_time],
                        self.time[sim_num][self.sim_time] - \
                                self.time[sim_num][self.sim_time-1], self.o_map[idx//self.samples_per_equation],
                    ))

            del self.data
            del self.tokens
            del self.grid
            del self.time
            gc.collect()
            print("TOTAL SAMPLES: {}".format(len(self.data_tuples)))
            print("Done.")

    def _encode_tokens(self, all_tokens):
        encoded_tokens = []
        num_concat = 0
        for i in range(len(all_tokens)):
            try: # All the operators, bcs, regular symbols
                encoded_tokens.append(self.word2id[all_tokens[i]])
                if(all_tokens[i] == "&"): # 5 concatenations before we get to lists of sampled values
                    num_concat += 1
            except KeyError: # Numerical values
                if(isinstance(all_tokens[i], str)):
                    for v in all_tokens[i]:
                        print(i, all_tokens[i])
                        try:
                            encoded_tokens.append(self.word2id[v])
                        except KeyError:
                            print(all_tokens)
                            raise
                    if(num_concat >= 5): # We're in a list of sampled parameters
                        encoded_tokens.append(self.word2id[","])
                else:
                    raise KeyError("Unrecognized token: {}".format(all_tokens[i]))
    
        return encoded_tokens

    def pretrain(self):
        self._pretrain = True

    def pretrain_off(self):
        self._pretrain = False

    def __len__(self):
        if(self.train_style == 'fixed_future'):
            if(self.split_style == 'equation'):
                print(len(self.available_idxs))
                return len(self.available_idxs)
            else:
                return len(self.data_tuples)
        elif(self.train_style == 'next_step'):
            if(self.split_style == 'equation'):
                return len(self.available_idxs)
            else:
                return len(self.data_tuples)
        elif(self.train_style == 'rollout'):
            return len(self.available_idxs)

    def __getitem__(self, idx):
        '''
        idx samples the file.
        Need to figure out a way to sample the snapshots within the file...
        '''
        if(self.split_style == 'initial_condition'):
            if(self._pretrain):
                return self.data_tuples[idx]
            else:
                return self.data_tuples[idx][:-1]
            idx = self.idx_to_avail_map[self.idxs[idx]]

        sim_idx = self.available_idxs[idx]
        sim_num = sim_idx // self.data.shape[1] # Get simulation number
        sim_time = sim_idx % self.data.shape[1] # Get time from that simulation
        if(self.train_style == "next_step"):
            if(self.return_text):
                return self.data[sim_num][sim_time-self.initial_step:sim_time], \
                        self.data[sim_num][sim_time][...,np.newaxis], \
                        self.grid[sim_num//2], \
                        self.tokens[sim_num//2][sim_time], \
                        self.time[sim_num//2][sim_time] - self.time[sim_num//2][sim_time-1]#, \
            else:
                return self.data[idx][...,:self.initial_step,:], \
                       self.data[idx][self.sim_time], \
                       self.grid[udx][self.sim_time]

        elif(self.train_style == 'fixed_future'):
            #print(self.time[0][:self.initial_step], self.time[0][self.sim_time])
            #raise
            if(self.return_text):
                return self.data[sim_num][:self.initial_step], \
                        self.data[sim_num][self.sim_time][...,np.newaxis], \
                        self.grid[sim_num], \
                        self.tokens[sim_num][self.sim_time], \
                        self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1]#, \
            else:
                return self.data[idx][:self.initial_step], \
                       self.data[idx][self.sim_time], \
                       self.grid[idx][self.sim_time]


class ElectricTransformerOperatorDataset2D(Dataset):
    def __init__(self, f, 
                 initial_step=10,
                 saved_folder='./data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 num_t=200,
                 num_x=200,
                 sim_time=-1,
                 split="train",
                 #test_ratio=0.2,
                 #val_ratio=0.2,
                 test_ratio=0.2,
                 val_ratio=0.2,
                 num_samples=None,
                 return_text=False,
                 train_style='fixed_future',
                 rollout_length=10,
                 split_style='equation',
                 seed=0
                 ):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """
        
        # Define path to files
        self.file_path = os.path.abspath(f.filename)
        self.return_text = return_text
        self.train_style = train_style
        self.rollout_length = rollout_length
        self.split_style = split_style
        
        # Extract list of seeds
        self.data_list = list(f.keys())

        # Time steps used as initial conditions
        self.initial_step = initial_step

        self.WORDS = ['(', ')', '[', ']', '+', '-', '*', '/', '=', 'Derivative', 'sin', 'cos', 't', 'u', 'x', 'w', 'y',
                 'pi', 'Delta', 'nabla', 'dot', "None", '0', '1', '2', '3', '4', '5', '6', '7', '8',
                 '9', '10^', 'E', 'e', ',', '.', '&', "Dirichlet", "Neumann"]
        self.word2id = {w: i for i, w in enumerate(self.WORDS)}
        self.id2word = {i: w for i, w in enumerate(self.WORDS)}

        self.num_t = num_t
        self.num_x = num_x

        self.h5_file = h5py.File(self.file_path, 'r')
        self.sim_time = sim_time

        sample_num = 0
        # Get all indices
        #TODO this shuffles by EQUATION, need to shuffle by SIMULATION?

        # Split indices into
        #print(idxs)
        #raise
        #TODO If using single BC combination, select at random for each run?
        print("\nSEED: {}".format(seed))
        np.random.seed(seed)
        if(self.split_style == 'equation'):
            np.random.shuffle(self.data_list)
            train_idx = int(num_samples * (1 - test_ratio - val_ratio))
            val_idx = int(num_samples * (1-test_ratio))
            if(split == "train"):
                self.data_list = self.data_list[:train_idx]
            elif(split == "val"):
                self.data_list = self.data_list[train_idx:val_idx]
            elif(split == "test"):
                self.data_list = self.data_list[val_idx:num_samples]
            else:
                raise ValueError("Select train, val, or test split. {} is invalid.".format(split))
        #print(self.data_list)
        #idxs = []
        #for i in tqdm(range(len(self.data_list))):
        #    seed_group = self.h5_file[self.data_list[i]]
        #    samples_per_sim = seed_group['b'].shape[0]
        #    for j in range(seed_group['b'].shape[0]):
        #        idxs.append(i*seed_group['b'].shape[0] + j)

        idxs = [i for i in range(len(self.data_list))] #TODO 
        np.random.shuffle(idxs)
        self.idxs = idxs[:num_samples]

        self.data = []
        self.grid = []
        self.time = []
        self.b = []
        self.temp_tokens = []
        self.available_idxs = []
        self.data_list = np.array(self.data_list)[self.idxs]
        #print(self.data_list)
        for i in tqdm(range(len(self.data_list))):
            seed_group = self.h5_file[self.data_list[i]]
            data = seed_group['v'][:]

            # Get extra info
            base_tokens = seed_group['tokens'][:]
            x = seed_group['X'][:][...,np.newaxis]
            y = seed_group['Y'][:][...,np.newaxis]
            b = seed_group['b'][:][...,np.newaxis]  #TODO Add this + time of 0 to data/tokens.

            # Add initial condition
            #complete_data = np.concatenate((w0, data), axis=3)
            self.data.append(data)
            self.b.append(b)
            #print(complete_data.shape)

            # Get grid
            self.grid.append(np.dstack((x,y)))

            # Get tokens
            self.temp_tokens.append(base_tokens)
            #print("\nGOT SAMPLE {}\n".format(i))

        # Arrange data
        self.data = torch.Tensor(np.array(self.data))

        # Get valid indices for returning data
        #print("Getting available idxs...")
        #self.available_idxs = []
        #for i in tqdm(range(len(self.data_list))):
            #if(self.train_style == 'next_step'):
            #    idxs = np.arange(0, self.data.shape[2])[self.initial_step:]
            #elif(self.train_style == 'arbitrary_step'):
            #    idxs = np.arange(0, self.data.shape[2])[self.initial_step:]
            
            # Take into account that the first self.initial_step samples can't be used as target
            #if(len(self.available_idxs) != 0): #TODO Make this robust to initial step
            #    idxs += self.available_idxs[-1] + 1 if(self.train_style == 'next_step') else \
            #            self.available_idxs[-1] + 1 + self.rollout_length if(self.train_style == 'rollout') else \
        #                                        self.available_idxs[-1] + 1
            #self.available_idxs.extend(idxs)


        # Grid to tensor
        self.grid = torch.Tensor(np.array(self.grid))

        # Add tokenized time to each equation for each simulation
        #print("Getting tokens...")
        self.tokens = []
        self.tokens = torch.empty(self.data.shape[0], 100)
        for idx, token in enumerate(self.temp_tokens):

            # Hold on to tokens
            full_tokens = copy.copy(list(token))
            full_tokens.extend([len(self.WORDS)]*(100 - len(full_tokens)))
            self.tokens[idx] = torch.Tensor(full_tokens)

        # Time and tokens to tensors
        #self.time = torch.Tensor(np.array(self.time))
        self.tokens = torch.Tensor(self.tokens)

        if(self.split_style == 'initial_condition'):
            train_idx = int(len(self.available_idxs) * (1 - test_ratio - val_ratio))
            val_idx = int(len(self.available_idxs) * (1-test_ratio))
            self.idxs = [i for i in range(len(self.available_idxs))]
            np.random.shuffle(self.idxs)
            #print(train_idx, val_idx, num_samples, len(self.idxs))
            if(split == "train"):
                self.idxs = self.idxs[:train_idx]
            elif(split == "val"):
                self.idxs = self.idxs[train_idx:val_idx]
            elif(split == "test"):
                self.idxs = self.idxs[val_idx:]
            else:
                raise ValueError("Select train, val, or test split. {} is invalid.".format(split))
            self.idx_to_avail_map = {i[0]: i[1] for i in zip(self.idxs, self.available_idxs)}
            self.sample_to_idx_map = {i[0]: i[1] for i in zip(self.idxs, self.available_idxs)}

        self.h5_file.close()
        print("Number of samples: {}".format(len(self.data)))
        print("Done.")

        # Create data tuples?
        self.data_tuples = []
        if(self.split_style == 'initial_condition'):
            for idx in range(len(self.idxs)):
                idx = self.idx_to_avail_map[self.idxs[idx]]

                sim_idx = self.available_idxs[idx]
                sim_num = sim_idx // self.data.shape[1] # Get simulation number
                sim_time = sim_idx % self.data.shape[1] # Get time from that simulation

                self.data_tuples.append((self.data[sim_num][sim_time-self.initial_step:sim_time],
                        self.data[sim_num][sim_time][...,np.newaxis],
                        self.grid[sim_num],
                        self.tokens[sim_num][sim_time],
                        self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1]))
            del self.data
            del self.tokens
            del self.grid
            del self.time
            gc.collect()
            #print(len(self.data_tuples))

    def _encode_tokens(self, all_tokens):
        encoded_tokens = []
        num_concat = 0
        for i in range(len(all_tokens)):
            try: # All the operators, bcs, regular symbols
                encoded_tokens.append(self.word2id[all_tokens[i]])
                if(all_tokens[i] == "&"): # 5 concatenations before we get to lists of sampled values
                    num_concat += 1
            except KeyError: # Numerical values
                if(isinstance(all_tokens[i], str)):
                    for v in all_tokens[i]:
                        print(i, all_tokens[i])
                        try:
                            encoded_tokens.append(self.word2id[v])
                        except KeyError:
                            print(all_tokens)
                            raise
                    if(num_concat >= 5): # We're in a list of sampled parameters
                        encoded_tokens.append(self.word2id[","])
                else:
                    raise KeyError("Unrecognized token: {}".format(all_tokens[i]))
    
        return encoded_tokens

    def __len__(self):
        if(self.train_style == 'fixed_future'):
            return len(self.data_list)
        elif(self.train_style == 'next_step'):
            if(self.split_style == 'equation'):
                return len(self.data)
            else:
                #return len(self.idxs)
                return len(self.data_tuples)
        elif(self.train_style == 'rollout'):
            return len(self.available_idxs)

    def __getitem__(self, idx):
        '''
        idx samples the file.
        Need to figure out a way to sample the snapshots within the file...
        '''

        if(self.return_text):
            return self.b[idx], \
                    self.data[idx][...,np.newaxis], \
                    self.grid[idx], \
                    self.tokens[idx], \
                    1.0


class Dataset2D(Dataset):
    def __init__(self, f, 
                 initial_step=10,
                 saved_folder='./data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 num_t=200,
                 num_x=200,
                 sim_time=-1,
                 split="train",
                 #test_ratio=0.2,
                 #val_ratio=0.2,
                 test_ratio=0.2,
                 val_ratio=0.2,
                 num_samples=None,
                 return_text=False,
                 train_style='fixed_future',
                 rollout_length=10,
                 split_style='equation',
                 seed=0,
                 pretraining=False
                 ):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """
        print("\nSEED: {}".format(seed))
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Define path to files
        self.file_path = os.path.abspath(f.filename)
        self.return_text = return_text
        self.train_style = train_style
        self.rollout_length = rollout_length
        self.split_style = split_style
        self.pretraining = pretraining
        
        # Extract list of seeds
        self.temp_data_list = list(f.keys())
        self.data_list = []
        #TODO: Take this out once on a cluster (may not matter right now?)
        for f in self.temp_data_list: # Can also use this to filter diffusion-only systems
            if(float(f.split("_")[1]) < 0.05):
                self.data_list.append(f)
        np.random.shuffle(self.data_list)
        self.data_list = self.data_list[:num_samples]

        # Time steps used as initial conditions
        self.initial_step = initial_step

        self.WORDS = ['(', ')', '+', '-', '*', '/', '=', 'Derivative', 'sin', 'cos', 't', 'u', 'x', 'w', 'v', 'y',
                 'pi', 'Delta', 'nabla', 'dot', "None", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                 '10^', 'E', 'e', ',', '.', '&']
        self.word2id = {w: i for i, w in enumerate(self.WORDS)}
        self.id2word = {i: w for i, w in enumerate(self.WORDS)}

        self.num_t = num_t
        self.num_x = num_x
        #self.name = "pde_{}-{}".format(self.num_t, self.num_x)

        self.h5_file = h5py.File(self.file_path, 'r')
        self.sim_time = sim_time

        sample_num = 0
        # Get all indices
        #TODO this shuffles by EQUATION, need to shuffle by SIMULATION?

        # Split indices into
        #TODO If using single BC combination, select at random for each run?
        if(self.split_style == 'equation'):
            np.random.shuffle(self.data_list)
            train_idx = int(num_samples * (1 - test_ratio - val_ratio))
            val_idx = int(num_samples * (1-test_ratio))
            if(split == "train"):
                self.data_list = self.data_list[:train_idx]
            elif(split == "val"):
                self.data_list = self.data_list[train_idx:val_idx]
            elif(split == "test"):
                self.data_list = self.data_list[val_idx:num_samples]
            else:
                raise ValueError("Select train, val, or test split. {} is invalid.".format(split))

        idxs = [i for i in range(len(self.data_list))] #TODO 
        #np.random.shuffle(idxs)
        self.idxs = idxs[:num_samples]

        #self.data = []
        #self.data = torch.empty(len(self.data_list), 101, 64, 64)
        self.data = torch.empty(len(self.data_list), 121, 64, 64)
        self.grid = torch.empty(len(self.data_list), 64, 64, 2)
        self.time = []
        self.temp_tokens = []
        self.alt_temp_tokens = []
        self.available_idxs = []
        self.all_operator_maps = []
        #print(self.data_list)
        #for f in self.data_list:
        #    print(f)
        #self.data_list = self.data_list[[float(t.split("_")[0]) <= 0.5 for t in self.data_list]]
        #self.data_list = np.array(self.data_list)[self.idxs]
        #print(self.data_list)
        #raise
        for i in tqdm(range(len(self.data_list))):

            # Get all data from file
            seed_group = self.h5_file[self.data_list[i]]
            ic = torch.tensor(seed_group['u0'][:]).float()
            data = torch.tensor(seed_group['u'][:][:self.sim_time]).float()
            grid = torch.tensor(seed_group['grid'][:]).float()

            #if(float(self.data_list[i].split("_")[0]) > 0.05):
            #    continue
            if(data.isnan().any()):
                print("\nFOUND A NAN\n")
                print(self.data_list[i])
                raise

            # Stack data
            all_data = torch.vstack((ic.unsqueeze(0), data))

            # Get tokens
            base_tokens = seed_group['tokens'][:]
            time = seed_group['time'][:]
            time = np.insert(time, 0, 0.)

            # Store all values
            self.data[i] = all_data
            self.grid[i] = grid
            self.temp_tokens.append(base_tokens)
            self.time.append(time)

            dl_split = self.data_list[i].split("_")
            if(dl_split[0] == 'Burgers'):
                o_map = [float(dl_split[1]), float(dl_split[2]), float(dl_split[3]), 0.]
            elif(dl_split[0] == 'Heat'):
                o_map = [float(dl_split[1]), 0., 0., 0.]
            self.all_operator_maps.append(o_map)

        # Add tokenized time to each equation for each simulation
        self.tokens = []
        self.tokens = torch.empty(self.data.shape[0], self.data.shape[1], 300)
        for idx, token in enumerate(self.temp_tokens):
            for jdx, time in enumerate(self.time[idx]):
                # Tokenize time
                slice_tokens = self._encode_tokens("&" + str(time))

                # Add tokenized time to equation
                full_tokens = copy.copy(list(token))
                full_tokens.extend(list(slice_tokens))

                # Hold on to tokens
                full_tokens = copy.copy(list(token))
                full_tokens.extend([len(self.WORDS)]*(300 - len(full_tokens)))
                self.tokens[idx][jdx] = torch.Tensor(full_tokens)

        # Time and tokens to tensors
        self.tokens = torch.Tensor(self.tokens)
        self.time = torch.Tensor(self.time)
        self.all_operator_maps = torch.Tensor(np.array(self.all_operator_maps))

        del self.temp_tokens

        if(self.train_style in ['next_step', 'arbitrary_step']):
            for i in tqdm(range(len(self.data_list))):
                if(self.train_style == 'next_step'):
                    idxs = np.arange(0, self.data.shape[1])[self.initial_step:]
                    #if(self.split_style == 'equation'):
                    #    idxs = np.append(idxs, np.arange(0, self.data.shape[1])[self.initial_step:] + idxs[-1]+1)
                elif(self.train_style == 'arbitrary_step'):
                    idxs = np.arange(0, self.data.shape[2])[self.initial_step:]
                
                # Take into account that the first self.initial_step samples can't be used as target
                if(len(self.available_idxs) != 0): #TODO Make this robust to initial step
                    idxs += self.available_idxs[-1] + 1 if(self.train_style == 'next_step') else \
                            self.available_idxs[-1] + 1 + self.rollout_length if(self.train_style == 'rollout') else \
	    					self.available_idxs[-1] + 1
                self.available_idxs.extend(idxs)

        elif(self.train_style == 'fixed_future'): # Only need to keep track of total number of valid samples
            idxs = np.arange(0, self.data.shape[0]*self.data.shape[1])
            self.available_idxs = idxs


        if(self.split_style == 'initial_condition'):
            train_idx = int(len(self.available_idxs) * (1 - test_ratio - val_ratio))
            val_idx = int(len(self.available_idxs) * (1-test_ratio))
            #print(len(self.available_idxs))
            self.idxs = [i for i in range(len(self.available_idxs))]
            #raise
            #np.random.shuffle(self.idxs)
            if(split == "train"):
                self.idxs = self.idxs[:train_idx]
            elif(split == "val"):
                self.idxs = self.idxs[train_idx:val_idx]
            elif(split == "test"):
                self.idxs = self.idxs[val_idx:]
            else:
                raise ValueError("Select train, val, or test split. {} is invalid.".format(split))
            self.idx_to_avail_map = {i[0]: i[1] for i in zip(self.idxs, self.available_idxs)}
            self.sample_to_idx_map = {i[0]: i[1] for i in zip(self.idxs, self.available_idxs)}

        self.h5_file.close()
        print("Number of samples: {}".format(len(self.data)))
        print("Available idxs: {}".format(len(self.available_idxs)))
        print("Done.")

        # Create data tuples?
        self.data_tuples = []
        #print(self.idxs)
        #print(self.data_list[-1])
        #print(self.tokens.shape)
        #print(self.tokens[-1][0])
        #print(self.data.shape)
        #print(self.data_list)
        if(self.split_style == 'initial_condition'):
            for idx in self.idxs:

                sim_idx = self.available_idxs[idx]
                sim_num = sim_idx // self.data.shape[1] # Get simulation number
                sim_time = sim_idx % self.data.shape[1] # Get time from that simulation
                #print(sim_num, sim_time)

                if(self.pretraining):
                    self.data_tuples.append((self.data[sim_num][sim_time-self.initial_step:sim_time],
                            self.data[sim_num][sim_time][...,np.newaxis],
                            self.grid[sim_num],
                            self.tokens[sim_num][sim_time],
                            self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1],
                            self.all_operator_maps[sim_num]))
                else:
                    self.data_tuples.append((self.data[sim_num][sim_time-self.initial_step:sim_time],
                            self.data[sim_num][sim_time][...,np.newaxis],
                            self.grid[sim_num],
                            self.tokens[sim_num][sim_time],
                            self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1]))
                #self.data_tuples.append((self.data[sim_num][sim_time-self.initial_step:sim_time],
                #        self.data[sim_num][sim_time][...,np.newaxis],
                #        self.grid[sim_num],
                #        self.tokens[sim_num][sim_time],
                #        self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1]))
            del self.data
            del self.tokens
            del self.grid
            del self.time
            torch.cuda.empty_cache()
            gc.collect()

        #raise

    def _encode_tokens(self, all_tokens):
        encoded_tokens = []
        num_concat = 0
        for i in range(len(all_tokens)):
            try: # All the operators, bcs, regular symbols
                encoded_tokens.append(self.word2id[all_tokens[i]])
                if(all_tokens[i] == "&"): # 5 concatenations before we get to lists of sampled values
                    num_concat += 1
            except KeyError: # Numerical values
                if(isinstance(all_tokens[i], str)):
                    for v in all_tokens[i]:
                        #print(i, all_tokens[i])
                        try:
                            encoded_tokens.append(self.word2id[v])
                        except KeyError:
                            #print(all_tokens)
                            raise
                    if(num_concat >= 5): # We're in a list of sampled parameters
                        encoded_tokens.append(self.word2id[","])
                else:
                    raise KeyError("Unrecognized token: {}".format(all_tokens[i]))
    
        return encoded_tokens


    def _one_hot_encode(self, tokens):
        encoding = np.zeros((len(tokens), len(self.WORDS)+1))
        encoding[range(tokens.shape[0]), tokens] = 1
        return encoding


    def __len__(self):
        if(self.train_style == 'fixed_future'):
            return len(self.data_list)
        elif(self.train_style == 'next_step'):
            if(self.split_style == 'equation'):
                return len(self.available_idxs)
            else:
                return len(self.data_tuples)
        elif(self.train_style == 'rollout'):
            return len(self.available_idxs)
    

    def __getitem__(self, idx):
        '''
        idx samples the file.
        Need to figure out a way to sample the snapshots within the file...
        '''
        if(self.split_style == 'initial_condition'):
            return self.data_tuples[idx]
            idx = self.idx_to_avail_map[self.idxs[idx]]

        sim_idx = self.available_idxs[idx]
        sim_num = sim_idx // self.data.shape[1] # Get simulation number
        sim_time = sim_idx % self.data.shape[1] # Get time from that simulation
        if(self.train_style == "next_step"):
            if(self.return_text):
                return  self.data[sim_num][sim_time-self.initial_step:sim_time], \
                        self.data[sim_num][sim_time][...,np.newaxis], \
                        self.grid[sim_num], \
                        self.tokens[sim_num], \
                        self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1]#, \
            else:
                return self.data[idx][...,:self.initial_step,:], \
                       self.data[idx][self.sim_time], \
                       self.grid[idx][self.sim_time]

        elif(self.train_style == 'fixed_future'):
            if(self.return_text):
                return self.data[sim_num][:self.initial_step], \
                        self.data[sim_num][self.sim_time][...,np.newaxis], \
                        self.grid[sim_num], \
                        self.tokens[sim_num][self.sim_time], \
                        self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1]#, \
            else:
                return self.data[idx][:self.initial_step], \
                       self.data[idx][self.sim_time], \
                       self.grid[idx][self.sim_time]
                       

class MultiDataset2D(Dataset):
    def __init__(self, 
                 initial_step=10,
                 saved_folder='./data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 num_t=200,
                 num_x=200,
                 sim_time=-1,
                 split="train",
                 #test_ratio=0.2,
                 #val_ratio=0.2,
                 test_ratio=0.2,
                 val_ratio=0.2,
                 num_samples=None,
                 return_text=False,
                 train_style='fixed_future',
                 rollout_length=10,
                 split_style='equation',
                 seed=0,
                 pretraining=False
                 ):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """
        
        # Define path to files
        #self.file_path = os.path.abspath(f.filename)
        self.return_text = return_text
        self.train_style = train_style
        self.rollout_length = rollout_length
        self.split_style = split_style
        self.pretraining = pretraining
        self._pretrain_tokens = False
        
        data_files = ['2d_Heat_30s_50inits_19nus_ns_match.h5', '2d_Burgers_30s_5inits_304systems_ns_match.h5']
        #data_files = ['2d_Burgers_5s_5inits_448systems.h5', '2d_Heat_5s_50inits_28nus.h5']
        #data_files = ['2d_Heat_5s_50inits_28nus.h5', '2d_Heat_5s_50inits_28nus.h5']
        

        # Time steps used as initial conditions
        self.initial_step = initial_step

        #self.WORDS = ['(', ')', '+', '-', '*', '/', '=', 'Derivative', 'sin', 'cos', 't', 'u', 'x', 'w', 'y',
        #              'pi', 'Delta', 'nabla', 'dot', "None", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10^',
        #              'E', 'e', ',', '.', '&', "Dirichlet", "Neumann"]
        #self.WORDS = ['(', ')', '[', ']', '+', '-', '*', '/', '=', 'Derivative', 'sin', 'cos', 't', 'u', 'x', 'w', 'y',
        #         'pi', 'Delta', 'nabla', 'dot', "None", '0', '1', '2', '3', '4', '5', '6', '7', '8',
        #         '9', '10^', 'E', 'e', ',', '.', '&', "Dirichlet", "Neumann"]
        self.WORDS = ['(', ')', '+', '-', '*', '/', '=', 'Derivative', 'sin', 'cos', 't', 'u', 'x', 'w', 'v', 'y',
                 'pi', 'Delta', 'nabla', 'dot', "None", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                 '10^', 'E', 'e', ',', '.', '&']
        self.word2id = {w: i for i, w in enumerate(self.WORDS)}
        self.id2word = {i: w for i, w in enumerate(self.WORDS)}

        self.num_t = num_t
        self.num_x = num_x
        #self.name = "pde_{}-{}".format(self.num_t, self.num_x)

        self.data = []
        self.grid = []
        self.time = []
        self.temp_tokens = []
        self.alt_temp_tokens = []
        self.available_idxs = []
        self.all_data_list = []
        self.all_operator_maps = []
        self.data_idxs = []
        for jdx, df in enumerate(data_files):
            np.random.seed(seed)
            torch.manual_seed(seed)

            h5_file = h5py.File("{}{}".format(saved_folder, df), 'r')

            # Extract list of seeds
            #self.data_list = list(h5_file.keys())
            self.temp_data_list = list(h5_file.keys())
            self.data_list = []
            #TODO: Take this out once on a cluster (may not matter right now?)
            for f in self.temp_data_list: # Can also use this to filter diffusion-only systems
                if(float(f.split("_")[1]) < 0.05):
                    self.data_list.append(f)
            np.random.shuffle(self.data_list)
            self.data_list = self.data_list[:num_samples]
            #print(self.data_list)
            self.sim_time = sim_time

            sample_num = 0
            # Get all indices
            #TODO this shuffles by EQUATION, need to shuffle by SIMULATION?

            # Split indices into
            #TODO If using single BC combination, select at random for each run?
            print("\nSEED: {}".format(seed))
            if(self.split_style == 'equation'):
                np.random.shuffle(self.data_list)
                train_idx = int(num_samples * (1 - test_ratio - val_ratio))
                val_idx = int(num_samples * (1-test_ratio))
                if(split == "train"):
                    self.data_list = self.data_list[:train_idx]
                elif(split == "val"):
                    self.data_list = self.data_list[train_idx:val_idx]
                elif(split == "test"):
                    self.data_list = self.data_list[val_idx:num_samples]
                else:
                    raise ValueError("Select train, val, or test split. {} is invalid.".format(split))
            #print(self.data_list)
            #idxs = []
            #for i in tqdm(range(len(self.data_list))):
            #    seed_group = self.h5_file[self.data_list[i]]
            #    samples_per_sim = seed_group['b'].shape[0]
            #    for j in range(seed_group['b'].shape[0]):
            #        idxs.append(i*seed_group['b'].shape[0] + j)

            idxs = [i for i in range(len(self.data_list))] #TODO 
            #np.random.shuffle(idxs)
            self.idxs = idxs[:num_samples]
            #self.data_list = np.array(self.data_list)[self.idxs]
            self.all_data_list.append(self.data_list)

            for i in tqdm(range(len(self.data_list))):
                # Get all data from file
                seed_group = h5_file[self.data_list[i]]
                ic = torch.tensor(seed_group['u0'][:]).float()
                data = torch.tensor(seed_group['u'][:][:self.sim_time]).float()
                #alpha = torch.tensor(seed_group['alpha'][:]).float()
                grid = torch.tensor(seed_group['grid'][:]).float()

                dl_split = self.data_list[i].split("_")
                #if(float(dl_split[0]) > 0.05 and len(dl_split) == 4):
                #    continue
                if(data.isnan().any()):
                    print("\nFOUND A NAN\n")
                    print(self.data_list[i])
                    raise

                # Stack data
                all_data = torch.vstack((ic.unsqueeze(0), data))

                # Get tokens
                base_tokens = seed_group['tokens'][:]
                time = seed_group['time'][:]
                time = np.insert(time, 0, 0.)

                # Store all values
                self.data.append(all_data.unsqueeze(0))
                self.grid.append(grid.unsqueeze(0))
                self.temp_tokens.append(base_tokens)
                self.time.append(time)

                if(dl_split[0] == 'Burgers'):
                    o_map = [float(dl_split[1]), float(dl_split[2]), float(dl_split[3])]
                elif(dl_split[0] == 'Heat'):
                    o_map = [float(dl_split[1]), 0., 0.]
                self.all_operator_maps.append(o_map)

            # Add tokenized time to each equation for each simulation
            self.tokens = torch.empty(len(self.data), len(self.data[0][0]), 300)
            for idx, token in enumerate(self.temp_tokens):
                for jdx, time in enumerate(self.time[idx]):
                    # Tokenize time
                    slice_tokens = self._encode_tokens("&" + str(time))

                    # Add tokenized time to equation
                    full_tokens = copy.copy(list(token))
                    full_tokens.extend(list(slice_tokens))

                    # Hold on to tokens
                    full_tokens = copy.copy(list(token))
                    full_tokens.extend([len(self.WORDS)]*(300 - len(full_tokens)))
                    self.tokens[idx][jdx] = torch.Tensor(full_tokens)

            # Time and tokens to tensors
            self.tokens = torch.Tensor(self.tokens)

            self. new_available_idxs = []
            if(self.train_style in ['next_step', 'arbitrary_step']):
                for i in tqdm(range(len(self.data_list))):
                    if(self.train_style == 'next_step'):
                        idxs = np.arange(0, len(self.data[0][0]))[self.initial_step:]
                    elif(self.train_style == 'arbitrary_step'):
                        idxs = np.arange(0, len(self.data[0][0][0]))[self.initial_step:]
                    
                    # Take into account that the first self.initial_step samples can't be used as target
                    if(len(self.available_idxs) != 0): #TODO Make this robust to initial step
                        idxs += self.available_idxs[-1] + 1 if(self.train_style == 'next_step') else \
                                self.available_idxs[-1] + 1 + self.rollout_length if(self.train_style == 'rollout') else \
	        					self.available_idxs[-1] + 1
                    self.available_idxs.extend(idxs)
                    self.new_available_idxs.extend(idxs)

            elif(self.train_style == 'fixed_future'): # Only need to keep track of total number of valid samples
                idxs = np.arange(0, len(self.data)*len(self.data[0]))
                self.available_idxs = idxs

            if(self.split_style == 'initial_condition'):
                train_idx = int(len(self.new_available_idxs) * (1 - test_ratio - val_ratio))
                val_idx = int(len(self.new_available_idxs) * (1-test_ratio))
                self.new_idxs = [i for i in range(len(self.new_available_idxs))]
                if(split == "train"):
                    self.new_idxs = self.new_idxs[:train_idx]
                elif(split == "val"):
                    self.new_idxs = self.new_idxs[train_idx:val_idx]
                elif(split == "test"):
                    self.new_idxs = self.new_idxs[val_idx:]
                else:
                    raise ValueError("Select train, val, or test split. {} is invalid.".format(split))
                #print(self.new_idxs)
                #print(self.data_idxs)
                self.new_idxs = np.array(self.new_idxs)
                if(len(self.data_idxs) != 0):
                    #print("\n\n\nSHOULD BE WORKIN??\n\n")
                    self.new_idxs += len(self.new_available_idxs)
                    #print(self.new_idxs)
                self.data_idxs.extend(self.new_idxs)
                print(len(self.available_idxs))

            h5_file.close()
        #self.idx_to_avail_map = {i[0]: i[1] for i in zip(self.idxs, self.available_idxs)}
        #self.sample_to_idx_map = {i[0]: i[1] for i in zip(self.idxs, self.available_idxs)}
        #print(test_ratio)
        #print(train_idx, val_idx)
        #print(len(self.new_idxs))
        #print(self.new_idxs[-1])
        #print(len(self.data_idxs))
        #print(self.data_idxs[-10:])

        # Convert to tensors
        self.data = torch.cat(self.data, dim=0)
        self.grid = torch.cat(self.grid, dim=0)
        self.time = torch.tensor(self.time)
        self.all_operator_maps = torch.Tensor(np.array(self.all_operator_maps))

        print("Number of samples: {}".format(len(self.data)))
        print("Available idxs: {}".format(len(self.available_idxs)))
        print("Done.")

        # Create data tuples?
        self.data_tuples = []
        #print(self.data_idxs)
        #raise
        #print(self.data_list[-1])
        #print(self.tokens.shape)
        #print(self.tokens[-1][0])
        #print(self.data.shape)
        #print(self.data_list)
        #raise
        if(self.split_style == 'initial_condition'):
            #for idx in range(len(self.idxs)):
            for idx in self.data_idxs:

                #print(len(self.available_idxs))
                #print(idx-len(self.new_available_idxs))
                #raise
                sim_idx = self.available_idxs[idx]
                sim_num = sim_idx // self.data.shape[1] # Get simulation number
                sim_time = sim_idx % self.data.shape[1] # Get time from that simulation
                #print(sim_num-100, sim_time)

                if(self.pretraining):
                    self.data_tuples.append((self.data[sim_num][sim_time-self.initial_step:sim_time],
                            self.data[sim_num][sim_time][...,np.newaxis],
                            self.grid[sim_num],
                            self.tokens[sim_num][sim_time],
                            self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1],
                            self.all_operator_maps[sim_num]))
                else:
                    self.data_tuples.append((self.data[sim_num][sim_time-self.initial_step:sim_time],
                            self.data[sim_num][sim_time][...,np.newaxis],
                            self.grid[sim_num],
                            self.tokens[sim_num][sim_time],
                            self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1]))
            del self.data
            del self.tokens
            del self.grid
            del self.time
            gc.collect()


    def _encode_tokens(self, all_tokens):
        encoded_tokens = []
        num_concat = 0
        for i in range(len(all_tokens)):
            try: # All the operators, bcs, regular symbols
                encoded_tokens.append(self.word2id[all_tokens[i]])
                if(all_tokens[i] == "&"): # 5 concatenations before we get to lists of sampled values
                    num_concat += 1
            except KeyError: # Numerical values
                if(isinstance(all_tokens[i], str)):
                    for v in all_tokens[i]:
                        #print(i, all_tokens[i])
                        try:
                            encoded_tokens.append(self.word2id[v])
                        except KeyError:
                            #print(all_tokens)
                            raise
                    if(num_concat >= 5): # We're in a list of sampled parameters
                        encoded_tokens.append(self.word2id[","])
                else:
                    raise KeyError("Unrecognized token: {}".format(all_tokens[i]))
    
        return encoded_tokens


    def _one_hot_encode(self, tokens):
        encoding = np.zeros((len(tokens), len(self.WORDS)+1))
        encoding[range(tokens.shape[0]), tokens] = 1
        return encoding


    def __len__(self):
        if(self.train_style == 'fixed_future'):
            #return len(self.data_list)
            return self.data.shape[0]
        elif(self.train_style == 'next_step'):
            if(self.split_style == 'equation'):
                return len(self.available_idxs)
            else:
                return len(self.data_tuples)
        elif(self.train_style == 'rollout'):
            return len(self.available_idxs)
    

    def pretrain(self):
        self._pretrain_tokens = True

    def pretrain_off(self):
        self._pretrain_tokens = False


    def __getitem__(self, idx):
        '''
        idx samples the file.
        Need to figure out a way to sample the snapshots within the file...
        '''
        if(self.split_style == 'initial_condition'):
            return self.data_tuples[idx]
            idx = self.idx_to_avail_map[self.idxs[idx]]

        sim_idx = self.available_idxs[idx]
        sim_num = sim_idx // self.data.shape[1] # Get simulation number
        sim_time = sim_idx % self.data.shape[1] # Get time from that simulation
        if(self.train_style == "next_step"):
            if(self.return_text):
                if(self._pretrain_tokens):
                    return  self.data[sim_num][sim_time-self.initial_step:sim_time], \
                            self.data[sim_num][sim_time][...,np.newaxis], \
                            self.grid[sim_num], \
                            self.tokens[sim_num], \
                            self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1], \
                            self.all_operator_maps[sim_num]
                else:
                    return  self.data[sim_num][sim_time-self.initial_step:sim_time], \
                            self.data[sim_num][sim_time][...,np.newaxis], \
                            self.grid[sim_num], \
                            self.tokens[sim_num], \
                            self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1]#, \
            else:

                return self.data[idx][...,:self.initial_step,:], \
                       self.data[idx][self.sim_time], \
                       self.grid[idx][self.sim_time]

        elif(self.train_style == 'fixed_future'):
            sim_num = idx
            if(self.return_text):
                if(self._pretrain_tokens):
                    return self.data[sim_num][:self.initial_step], \
                            self.data[sim_num][self.sim_time][...,np.newaxis], \
                            self.grid[sim_num], \
                            self.tokens[sim_num][self.sim_time], \
                            self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1], \
                            self.all_operator_maps[sim_num]
                else:
                    return self.data[sim_num][:self.initial_step], \
                            self.data[sim_num][self.sim_time][...,np.newaxis], \
                            self.grid[sim_num], \
                            self.tokens[sim_num][self.sim_time], \
                            self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1]#, \
            else:
                return self.data[sim_num][:self.initial_step], \
                       self.data[sim_num][self.sim_time], \
                       self.grid[sim_num]


class WrapperMultiDataset2D(Dataset):
    def __init__(self, 
                 initial_step=10,
                 saved_folder='./data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 num_t=200,
                 num_x=200,
                 sim_time=-1,
                 split="train",
                 #test_ratio=0.2,
                 #val_ratio=0.2,
                 test_ratio=0.2,
                 val_ratio=0.2,
                 num_samples=None,
                 return_text=False,
                 train_style='fixed_future',
                 rollout_length=10,
                 split_style='equation',
                 seed=0,
                 pretraining=False,
                 samples_per_equation=None,
                 ):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """
        
        # Define path to files
        #self.file_path = os.path.abspath(f.filename)
        self.return_text = return_text
        self.train_style = train_style
        self.rollout_length = rollout_length
        self.split_style = split_style
        self.pretraining = pretraining
        
        data_files = ['2d_Heat_30s_50inits_19nus_ns_match.h5',
                      '2d_Burgers_30s_5inits_304systems_ns_match.h5',
                      '2d_ns_vel_30s_5N_256_95eq.h5']
        

        # Time steps used as initial conditions
        self.initial_step = initial_step

        ##TODO: REPUPLOAD NS DATA BECAUSE WORD LIST WASN'T THE SAME
        self.WORDS = ['(', ')', '+', '-', '*', '/', '=', 'Derivative', 'sin', 'cos', 't', 'u', 'x', 'w', 'v', 'y',
                 'pi', 'Delta', 'nabla', 'dot', "None", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                 '10^', 'E', 'e', ',', '.', '&']
             #WORDS = ['(', ')', '+', '-', '*', '/', '=', 'Derivative', 'sin', 'cos', 't', 'u', 'x', 'w', 'y',
             #    'pi', 'Delta', 'nabla', 'dot', "None", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             #    '10^', 'E', 'e', ',', '.', '&']
        self.word2id = {w: i for i, w in enumerate(self.WORDS)}
        self.id2word = {i: w for i, w in enumerate(self.WORDS)}

        self.num_t = num_t
        self.num_x = num_x
        #self.name = "pde_{}-{}".format(self.num_t, self.num_x)

        self.datasets = []
        for df in data_files:
            print("\nDATASET: {}".format(df))
            h5_file = h5py.File("{}{}".format(saved_folder, df), 'r')
            if('Heat' in df or 'Burgers' in df):
                self.datasets.append(Dataset2D(h5_file,
                                        split=split,
                                        initial_step=initial_step,
                                        reduced_resolution=reduced_resolution,
                                        reduced_resolution_t=reduced_resolution_t,
                                        reduced_batch=reduced_batch,
                                        saved_folder=saved_folder,
                                        return_text=return_text,
                                        num_t=num_t,
                                        num_x=num_x,
                                        sim_time=sim_time,
                                        num_samples=num_samples,
                                        train_style=train_style,
                                        split_style=split_style,
                                        seed=seed,
                                        pretraining=self.pretraining
                ))
            elif('ns_vel' in df):
                self.datasets.append(TransformerOperatorDataset2D(h5_file,
                                        split=split,
                                        initial_step=initial_step,
                                        reduced_resolution=reduced_resolution,
                                        reduced_resolution_t=reduced_resolution_t,
                                        reduced_batch=reduced_batch,
                                        saved_folder=saved_folder,
                                        return_text=return_text,
                                        num_t=num_t,
                                        num_x=num_x,
                                        sim_time=sim_time,
                                        num_samples=num_samples,
                                        train_style=train_style,
                                        split_style=split_style,
                                        samples_per_equation=samples_per_equation,
                                        seed=seed, pad_length=300,
                                        pretrain=self.pretraining
                ))


    def _encode_tokens(self, all_tokens):
        encoded_tokens = []
        num_concat = 0
        for i in range(len(all_tokens)):
            try: # All the operators, bcs, regular symbols
                encoded_tokens.append(self.word2id[all_tokens[i]])
                if(all_tokens[i] == "&"): # 5 concatenations before we get to lists of sampled values
                    num_concat += 1
            except KeyError: # Numerical values
                if(isinstance(all_tokens[i], str)):
                    for v in all_tokens[i]:
                        #print(i, all_tokens[i])
                        try:
                            encoded_tokens.append(self.word2id[v])
                        except KeyError:
                            #print(all_tokens)
                            raise
                    if(num_concat >= 5): # We're in a list of sampled parameters
                        encoded_tokens.append(self.word2id[","])
                else:
                    raise KeyError("Unrecognized token: {}".format(all_tokens[i]))
    
        return encoded_tokens


    def _one_hot_encode(self, tokens):
        encoding = np.zeros((len(tokens), len(self.WORDS)+1))
        encoding[range(tokens.shape[0]), tokens] = 1
        return encoding


    def __len__(self):
        if(self.train_style == 'fixed_future'):
            return len(self.data_list)
        elif(self.train_style == 'next_step'):
            if(self.split_style == 'equation'):
                return len(self.available_idxs)
            else:
                #return len(self.data_tuples)
                return sum([len(ds.data_tuples) for ds in self.datasets])
        elif(self.train_style == 'rollout'):
            return len(self.available_idxs)
    

    def __getitem__(self, idx):
        '''
        idx samples the file.
        Need to figure out a way to sample the snapshots within the file...
        '''
        if(self.split_style == 'initial_condition'):
            didx = int(idx//(self.__len__()/len(self.datasets)))
            sidx = int(idx%(self.__len__()/len(self.datasets)))
            return self.datasets[didx].data_tuples[sidx]
        else:
            raise NotImplementedError


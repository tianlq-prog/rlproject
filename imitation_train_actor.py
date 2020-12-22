from ipdb import set_trace
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import numpy as np
from sklearn.utils import shuffle
from agent import A2CNet
import csv
import json
from sklearn.model_selection import train_test_split
import argparse
import os 
parser=argparse.ArgumentParser()
parser.add_argument('--gpu')
args=parser.parse_args()
if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
device = torch.device("cuda:0" if (args.gpu is not None) and torch.cuda.is_available() else "cpu")

def accuracy(ys, ts):
    correct_prediction = torch.eq(torch.max(ys, 1)[1], ts)
    return torch.mean(correct_prediction.float())

def dump_tensors(gpu_only=True):
	"""Prints a list of the Tensors being tracked by the garbage collector."""
	import gc
	total_size = 0
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj):
				if not gpu_only or obj.is_cuda:
					print("%s:%s%s %s" % (type(obj).__name__, 
										  " GPU" if obj.is_cuda else "",
										  " pinned" if obj.is_pinned else "",
										  pretty_size(obj.size())))
					total_size += obj.numel()
			elif hasattr(obj, "data") and torch.is_tensor(obj.data):
				if not gpu_only or obj.is_cuda:
					print("%s → %s:%s%s%s%s %s" % (type(obj).__name__, 
												   type(obj.data).__name__, 
												   " GPU" if obj.is_cuda else "",
												   " pinned" if obj.data.is_pinned else "",
												   " grad" if obj.requires_grad else "", 
												   " volatile" if obj.volatile else "",
												   pretty_size(obj.data.size())))
					total_size += obj.data.numel()
		except Exception as e:
			pass        
	print("Total size:", total_size)

if __name__ == '__main__':
    num_actions = 6
    batch_size = 5
    num_epochs = 100
    train_size = 100000000000
    model = A2CNet(args.gpu is not None)
    if args.gpu is not None:
        model=model.cuda()

    states_obs = []
    states_raw = []
    actions = []
    logFile_actions = 'simpleAgentActions_sequence_rawObs.txt'
    logFile_states_obs = 'simpleAgentStates_obs.txt'
    logFile_states_raw = 'simpleAgentStates_raw.txt'
    with open(logFile_actions,'r') as fp:
        re = csv.reader(fp, dialect='excel')
        for i,line in enumerate(re):
            if len(line)>0 and i<train_size:
                line = [int(w) for w in line]
                actions.append(line)

    with open(logFile_states_obs,'r') as fp:
        re = csv.reader(fp, dialect='excel')
        for i,line in enumerate(re):
            if len(line)>0 and i<train_size:
                line = [json.loads(l) for l in line]
                line = [[np.asarray(l)] for l in line]
                states_obs.append(line)

    with open(logFile_states_raw,'r') as fp:
        re = csv.reader(fp, dialect='excel')
        for i,line in enumerate(re):
            if len(line)>0 and i<train_size:
                line = [json.loads(l) for l in line]
                states_raw.append(np.asarray(line))

    n = len(actions)
    actions = torch.from_numpy((np.array(actions,dtype=np.long)))
    states_obs = torch.from_numpy(np.asarray(states_obs,dtype=np.float32))
    states_raw = torch.from_numpy(np.asarray(states_raw,dtype=np.float32))
  
    nr_train = int(0.9*n//1)
    nr_val = int(n-nr_train)

    perm_ind = torch.randperm(n)
    train_ind = perm_ind[0:nr_train]
    val_ind = perm_ind[nr_train:]

    X1_tr = states_obs[train_ind].to(device)
    X2_tr = states_raw[train_ind].to(device)
    y_tr = actions[train_ind].to(device)
    X1_val = states_obs[val_ind].to(device)
    X2_val = states_raw[val_ind].to(device)
    y_val = actions[val_ind].to(device)

    print('train size {}'.format(nr_train))
    print('validation size {}'.format(nr_val))

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    with open('imitation_actor_batch.log','w') as f:
        f.write('loss,acc\n')
    with open('imitation_actor_val.log','w') as f:
        f.write('loss,acc\n')
    # training loop
    for e in range(num_epochs):
        permutation = torch.randperm(nr_train)
        for i in range(0,nr_train, batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            tr_input1_batch, tr_input2_batch, tr_targets_batch = X1_tr[indices].squeeze(2), X2_tr[indices].squeeze(2), y_tr[indices].long()

            siz = tr_targets_batch.size()

            hn1_batch,cn1_batch = model.get_lstm_reset(siz[0])
            
            optimizer.zero_grad()
            tr_output,_,_,_ = model(tr_input1_batch,tr_input2_batch,hn1_batch,cn1_batch)
            tr_output = tr_output.permute(0,2,1) 
            batch_loss =criterion(tr_output, tr_targets_batch)

            batch_loss.backward()
            optimizer.step()
            train_acc = accuracy(tr_output, tr_targets_batch)
    
            # store training loss
            train_losses.append(batch_loss.item())
            train_accs.append(train_acc.item())
            with open('imitation_actor_batch.log','a') as f:
                f.write('%.4f,%.4f\n'%(train_losses[-1],train_accs[-1]))
            print(train_losses[-1],train_accs[-1])
            del tr_input1_batch
            del tr_input2_batch
            del tr_targets_batch
            del tr_output
            del batch_loss
            del hn1_batch
            del cn1_batch
            del train_acc
            model.zero_grad()
    
        if args.gpu is not None:
            dump_tensors()
            torch.cuda.empty_cache()
        model.eval()
        va = []
        bl = 0
        permutation = torch.randperm(nr_val)
        with torch.no_grad():
            for i in range(0,nr_val, batch_size):
                indices = permutation[i:i+batch_size]
                val_input1_batch, val_input2_batch, val_targets_batch = X1_val[indices].squeeze(2), X2_val[indices].squeeze(2), y_val[indices].long()
                # get validation input and expected output as torch Variables and make sure type is correct

                siz = val_targets_batch.size()
                hn1_batch,cn1_batch = model.get_lstm_reset(siz[0])

                # predict with validation input
                val_output,_,_,_ = model(val_input1_batch,val_input2_batch,hn1_batch,cn1_batch)
                val_output = val_output.permute(0,2,1) #switch dimensions for criterion
                val_acc = accuracy(val_output, val_targets_batch)
                batch_loss =criterion(val_output, val_targets_batch)
                va.append(val_acc.item())
                bl += batch_loss.item()
                del val_input1_batch
                del val_input2_batch
                del val_targets_batch
                del val_output
                del batch_loss
                del hn1_batch
                del cn1_batch
                del val_acc
                if args.gpu is not None:
                    torch.cuda.empty_cache()
    
        torch.save(model.state_dict(), 'model/imitation_actor.pth')
        model.train()
    
        # store loss and accuracy
        val_losses.append(bl)
        val_accs.append(np.mean(va))
        with open('imitation_actor_val.log','a') as f:
            f.write('%.4f,%.4f\n'%(val_losses[-1],val_accs[-1]))
    
        #if e % 10 == 0:
        print("Epoch %i, "
                "Train Cost: %0.3f"
                "\tVal Cost: %0.3f"
                "\t Val acc: %0.3f" % (e, 
                                        train_losses[-1],
                                        val_losses[-1],
                                        val_accs[-1]))


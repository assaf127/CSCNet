import dataloaders
from model.modules import ConvLista_T, ListaParams
import torch
import numpy as np
from tqdm import tqdm
import argparse
import uuid
import json
import os

# For speed-up
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("--stride", type=int, dest="stride", help="stride size", default=8)
parser.add_argument("--num-filters", type=int, dest="num_filters", help="Number of filters", default=175)
parser.add_argument("--kernel-size", type=int, dest="kernel_size", help="The size of the kernel", default=11)
parser.add_argument("--threshold", type=float, dest="threshold", help="Init threshold value", default=0.01)
parser.add_argument("--noise-level", type=int, dest="noise_level", help="Should be an int in the range [0,255]", default=25)
parser.add_argument("--lr", type=float, dest="lr", help="ADAM Learning rate", default=2e-4)
parser.add_argument("--lr-step", type=int, dest="lr_step", help="Learning rate decrease step", default=50)
parser.add_argument("--lr-decay", type=float, dest="lr_decay", help="ADAM Learning rate decay (on step)", default=0.35)
parser.add_argument("--eps", type=float, dest="eps", help="ADAM epsilon parameter", default=1e-3)
parser.add_argument("--unfoldings", type=int, dest="unfoldings", help="Number of LISTA unfoldings", default=12)
parser.add_argument("--num-epochs", type=int, dest="num_epochs", help="Total number of epochs to train", default=250)
parser.add_argument("--crop-size", type=int, dest="crop_size", help="Total number of epochs to train", default=128)
parser.add_argument("--out-dir", type=str, dest="out_dir", help="Results' dir path", default='trained_models')
parser.add_argument("--model-name", type=str, dest="model_name", help="The name of the model to be saved.", default=None)
parser.add_argument("--data-path", type=str, dest="data_path", help="Path to the dir containing the training and testing datasets.", default="./datasets/")
parser.add_argument("--batch-size", type=int, dest="batch_size", help="Number of images in a batch", default=1)
parser.add_argument("--scale-levels", type=int, dest="scale_levels", help="Number of scale levels to clean at", default=2)
parser.add_argument("--num-supports-train", type=int, dest="num_supports_train", help="Number of supports to sample for training", default=100)
parser.add_argument("--num-supports-eval", type=int, dest="num_supports_eval", help="Number of supports to sample for testing", default=100)
args = parser.parse_args()

args.test_path = [f'{args.data_path}/BSD68/']
args.train_path = [f'{args.data_path}/CBSD432/', f'{args.data_path}/waterloo/']
args.noise_std = args.noise_level / 255
args.guid = args.model_name if args.model_name is not None else uuid.uuid4()

params = ListaParams(args.kernel_size, args.num_filters, args.stride, args.unfoldings,
                     args.scale_levels, args.num_supports_train, args.num_supports_eval)
loaders = dataloaders.get_dataloaders(args.train_path, args.test_path, args.crop_size, args.batch_size)
model = ConvLista_T(params).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)

psnr = {x: np.zeros(args.num_epochs) for x in ['train', 'test']}

print(args.__dict__)
config_filename = f'{args.out_dir}/{args.guid}_config.json'
os.makedirs(os.path.dirname(config_filename), exist_ok=True)
with open(config_filename,'w') as json_file:
    json.dump(args.__dict__, json_file, sort_keys=True, indent=4)

print('Training model...')
for epoch in tqdm(range(args.num_epochs), position=0, leave=False):
    for phase in ['train', 'test']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        # Iterate over data.
        num_iters = 0
        for batch in tqdm(loaders[phase], position=1, leave=False):
            batch = batch.cuda()
            noise = torch.randn_like(batch) * args.noise_std
            noisy_batch = batch + noise

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                output = model(noisy_batch)
                loss = (output - batch).pow(2).sum() / batch.shape[0]

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            psnr[phase][epoch] += -10*np.log10(loss.item() / (batch.shape[2]*batch.shape[3]))
            num_iters += 1
        if phase == 'train':
            scheduler.step()

        psnr[phase][epoch] /= num_iters
        tqdm.write(f'{epoch}: {phase} PSNR: {psnr[phase][epoch]}')
        with open(f'{args.out_dir}/{args.guid}_{phase}.psnr','a') as psnr_file:
            psnr_file.write(f'{psnr[phase][epoch]},')
    # deep copy the model
    torch.save(model.state_dict(), f'{args.out_dir}/{args.guid}.model')

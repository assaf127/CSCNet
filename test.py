import torch
from model import utils, modules
import dataloaders
from os.path import join
from uuid import UUID
from matplotlib import pyplot as plt
import numpy as np
import argparse
import json
from tqdm import tqdm


# For speed-up
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", type=str, dest="root_folder", help="The trained model's dir path", default='./trained_models')
parser.add_argument("--model-name", type=str, dest="uuid", help="The model's name", required=True)
parser.add_argument("--data-path", type=str, dest="data_path", help="Path to the dir containing the training and testing datasets.", default="./datasets/")
parser.add_argument("--all-supports", action='store_true', dest="use_all_supports", help="Use all possible supports", default=False)
args = parser.parse_args()

config_filename = args.uuid + '_config.json'  # args.uuid+'.config'
model_filename = args.uuid + '.model'
config_path = join(args.root_folder, config_filename)

# with open(config_path) as conf_file:
#     conf = conf_file.read()
# conf = eval(conf)
with open(config_path, 'r') as conf_file:
    conf = json.load(conf_file)

params = modules.ListaParams(conf['kernel_size'], conf['num_filters'], conf['stride'], conf['unfoldings'],
                             conf['scale_levels'], conf['num_supports_train'], conf['num_supports_eval'])
model = modules.ConvLista_T(params)
model.load_state_dict(torch.load(join(args.root_folder, model_filename)))

test_path = [f'{args.data_path}/Set12/']
# test_path = [f'{args.data_path}/BSD68/']
loaders = dataloaders.get_dataloaders(test_path, test_path, 128, 1)
loaders['test'].dataset.verbose = True
model.eval()   # Set model to evaluate mode
model.cuda()

num_iters = 0
noise_std = conf['noise_std']
psnr = 0
print(f"Testing model: {args.uuid} with noise_std {noise_std*255} on test images...")
for batch, imagename in tqdm(loaders['test']):
    batch = batch.cuda()
    noise = torch.randn_like(batch) * noise_std
    noisy_batch = batch + noise

    # forward
    # track history if only in train
    with torch.set_grad_enabled(False):
        output = model(noisy_batch, all_supports=args.use_all_supports)
        loss = (output - batch).pow(2).sum() / batch.shape[0]

    # statistics
    cur_mse = -10*np.log10(loss.item() / (batch.shape[2]*batch.shape[3]))
    tqdm.write(f'{imagename[0]}:\t{cur_mse}')
    psnr += cur_mse
    num_iters += 1
print('===========================')
print(f'Average:\t{psnr/num_iters}')
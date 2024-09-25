import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
from datasets.Internet_Dataloader import TestData_for_Internet, TestData_for_RERAIN, FlexibleDataset
from utils import *
from utils.utils import *
from skimage.metrics import structural_similarity as compare_ssim
from numpy import *
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='FADformer', type=str, help='model name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--exp', default='rain200', type=str, help='experiment setting')
args = parser.parse_args()

def test(val_loader_full, network, result_dir):

    torch.cuda.empty_cache()

    network.eval()

    os.makedirs(result_dir, exist_ok=True)

    for batch in val_loader_full:

        source_img = batch['source'].cuda()
        file_name = batch['filename'][0]

        # Pad the input if not_multiple_of 8
        img_multiple_of = 4
        height, width = source_img.shape[2], source_img.shape[3]
        H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                    (width + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0
        source_img = F.pad(source_img, (0, padw, 0, padh), 'reflect')

        with torch.no_grad():
            output = network(source_img).clamp_(0, 1)
            # print(output.shape)

        # Unpad the output
        output = output[:, :, :height, :width]
        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(result_dir, file_name.split('.')[0] + '.png'), out_img)


if __name__ == '__main__':

    device_index = [0]
    network = eval(args.model)()
    network = nn.DataParallel(network, device_ids=device_index).cuda()
    network.load_state_dict(torch.load(
        './pretrain_weights/rain200H/FADformer_Rain200H.pth')[
                                'state_dict'])

    root_dir = './demo/input'
    result_dir = './demo/output'
    test_dataset = FlexibleDataset(root_dir)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=True,
                             num_workers=args.num_workers,
                             pin_memory=True)

    test(test_loader, network, result_dir)
# coding:utf-8
# By Yuxiang Sun, Aug. 2, 2019
# Email: sun.yuxiang@outlook.com

import os, shutil, stat
import argparse 
import numpy as np
import sys
from PIL import Image
import torch 
from torch.autograd import Variable
from util.util import visualize
from model import RTFNet  

n_class = 9
image_dir = './dataset/images/'
model_dir = './weights_backup/'

def main():

    model = eval(args.model_name)(n_class=n_class)
    if args.gpu >= 0: model.cuda(args.gpu)
    print('| loading model file %s... ' % model_file) 
    pretrained_weight = torch.load(model_file, map_location = lambda storage, loc: storage.cuda(args.gpu))
    own_state = model.state_dict()
    for name, param in pretrained_weight.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param) 
    print('done!')

    files = os.listdir(image_dir)
    for file in files:
        if file[-3:] != 'png': continue
        if 'flip' in file: continue
        image = np.asarray(Image.open(image_dir+file))
        image = torch.from_numpy(image).float()
        image.unsqueeze_(0)
        #print image.shape  # (1, 480, 640, 4)
        image = np.asarray(image, dtype=np.float32).transpose((0,3,1,2))/255.0
        #print image.shape  # (1, 4, 480, 640)

        image = Variable(torch.tensor(image))
        if args.gpu >= 0: image = image.cuda(args.gpu)

        model.eval()
        with torch.no_grad():
            logits = model(image)
            predictions = logits.argmax(1)
            visualize(file, predictions, args.weight_name)

        print('| %s:%s, prediction of %s has been saved in ./demo_results' %(args.model_name, args.weight_name, file))

if __name__ == "__main__": 

    if os.path.exists('./demo_results') is True:
        print('| previous \'./demo_results\' exist, will delete the folder')
        shutil.rmtree('./demo_results')
        os.makedirs('./demo_results')
        os.chmod('./demo_results', stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine
    else:
        os.makedirs('./demo_results')
        os.chmod('./demo_results', stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine

    parser = argparse.ArgumentParser(description='Run demo with pytorch')
    parser.add_argument('--model_name', '-M',  type=str, default='RTFNet')
    parser.add_argument('--weight_name', '-W', type=str, default='RTFNet_152')  # RTFNet_152, RTFNet_50, please change the number of layers in the network file
    parser.add_argument('--gpu',        '-G',  type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    print("\n| the gpu count:", torch.cuda.device_count())
    print("| the current used gpu:", torch.cuda.current_device(), '\n')

    model_dir = os.path.join(model_dir, args.weight_name)  # model_dir = './weights_backup/'

    if os.path.exists(model_dir) is False:
        print("| the %s does not exit." %(model_dir))
        sys.exit()
    model_file = os.path.join(model_dir, 'final.pth')
    if os.path.exists(model_file) is True:
        print('| use the final model file.')
    else:
        print('| no model file found.')
        sys.exit()
    print('| running %s:%s demo on GPU #%d with pytorch' % (args.model_name, args.weight_name, args.gpu))
    main()

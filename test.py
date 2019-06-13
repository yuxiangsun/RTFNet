# coding:utf-8
# modified from: https://github.com/haqishen/MFNet-pytorch
# By Yuxiang Sun, Nov. 8, 2018
# Email: sun.yuxiang@outlook.com

import os
import argparse
import time
import datetime
import numpy as np
import sys
import torch 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util.MF_dataset import MF_dataset 
from model import RTFNet 
from sklearn.metrics import confusion_matrix
  
n_class   = 9
data_dir  = './dataset/'
model_dir = './weights_backup/' 

def main():
   
    conf_total = np.zeros((n_class,n_class))

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

    test_dataset  = MF_dataset(data_dir, args.dataset_name, have_label=True, input_h=args.img_height, input_w=args.img_width)
    test_loader  = DataLoader(
        dataset     = test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    test_loader.n_iter = len(test_loader)
    ave_time_cost = 0.0

    model.eval()
 
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images)
            labels = Variable(labels)
            if args.gpu >= 0:
                images = images.cuda(args.gpu)
                labels = labels.cuda(args.gpu)

            start_time = time.time()
            logits = model(images)  # logits.size(): mini_batch*num_class*480*640
            end_time = time.time()
            if it>10: # # ignore the first 10 frames
                ave_time_cost += (end_time-start_time)

            # convert tensor to numpy 1d array
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
            # generate confusion matrix frame-by-frame
            conf = confusion_matrix(label, prediction, [0,1,2,3,4,5,6,7,8]) # conf is an n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            print("| frame %d/%d, time cost: %.2f ms" %(it+1, test_loader.n_iter, (end_time-start_time)*1000)) 
 
    # calculate recall (Acc) and IoU for each class 
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    for cid in range(0, n_class): # cid: class id 
        if conf_total[cid, 0:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, 0:].sum()) # recall (Acc) = TP/TP+FN
        if (conf_total[cid, 0:].sum() + conf_total[0:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, 0:].sum() + conf_total[0:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN
 
    print('\n###########################################################################')
    print('\n| %s: %s test results (with batch size %d) on %s using %s:' %(args.model_name, args.weight_name, batch_size, datetime.date.today(), torch.cuda.get_device_name(args.gpu))) 
    print('\n| * the tested dataset name: %s' % args.dataset_name)
    print('| * the tested image count: %d' % test_loader.n_iter)
    print('| * the tested image size: %d*%d' %(args.img_height, args.img_width)) 
    print("| * recall per class: \n    unlabeled: %.6f, car: %.6f, person: %.6f, bike: %.6f, curve: %.6f, car_stop: %.6f, guardrail: %.6f, color_cone: %.6f, bump: %.6f" \
          %(recall_per_class[0], recall_per_class[1], recall_per_class[2], recall_per_class[3], recall_per_class[4], recall_per_class[5], recall_per_class[6], recall_per_class[7], recall_per_class[8]))
    print("| * iou per class: \n    unlabeled: %.6f, car: %.6f, person: %.6f, bike: %.6f, curve: %.6f, car_stop: %.6f, guardrail: %.6f, color_cone: %.6f, bump: %.6f" \
          %(iou_per_class[0], iou_per_class[1], iou_per_class[2], iou_per_class[3], iou_per_class[4], iou_per_class[5], iou_per_class[6], iou_per_class[7], iou_per_class[8])) 

    print("\n| * average values (np.mean(x)): \n recall: %.6f, iou: %.6f" \
          %(recall_per_class.mean(), iou_per_class.mean()))
    print("| * average values (np.mean(np.nan_to_num(x))): \n recall: %.6f, iou: %.6f" \
          %(np.mean(np.nan_to_num(recall_per_class)), np.mean(np.nan_to_num(iou_per_class))))

    print('\n| * the average time cost per frame (with batch size %d): %.2f ms, namely, the inference speed is %.2f fps' %(batch_size, ave_time_cost*1000/(test_loader.n_iter-11), 1.0/(ave_time_cost/(test_loader.n_iter-11)))) # ignore the first 10 frames

    #print('\n| * the total confusion matrix: ') 
    #np.set_printoptions(precision=8, threshold=np.inf, linewidth=np.inf, suppress=True)
    #print(conf_total)
    print('\n###########################################################################')

if __name__ == '__main__':
 
    parser = argparse.ArgumentParser(description='Test with pytorch')
    parser.add_argument('--model_name',  '-M',  type=str, default='RTFNet')
    parser.add_argument('--weight_name', '-W', type=str, default='RTFNet_152') # RTFNet_152, RTFNet_50, please change the number of layers in the network file
    parser.add_argument('--dataset_name', '-D', type=str, default='test') # test, test_day, test_night
    parser.add_argument('--img_height', '-IH', type=int, default=480) 
    parser.add_argument('--img_width', '-IW', type=int, default=640)  
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=8)
    args = parser.parse_args()

    batch_size = 1 # do not change this parameter!	

    torch.cuda.set_device(args.gpu)
    print "\n| the gpu count:", torch.cuda.device_count()
    print "| the current used gpu:", torch.cuda.current_device(), '\n'

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
    print('| testing %s: %s on GPU #%d with pytorch' % (args.model_name, args.weight_name, args.gpu))
    main()

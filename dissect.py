#import settings
import argparse, os, shutil, time, warnings, datetime, sys

from feature_operation import hook_feature,FeatureOperator
from visualize.report import generate_html_summary
from util.clean import clean
import torch
import torchvision

def get_parser():
    parser = argparse.ArgumentParser(description='Netdissect configuration')
    parser.add_argument('MODEL_FILE', metavar='FILE',
                        help='path to dataset')
    parser.add_argument('OUTPUT_FOLDER', metavar='OUTPUT',
                        help='path to dataset')
    parser.add_argument('-d', '--DATASET', default='places365', type=str, 
                        help='model trained on: places365 or imagenet (default: places365)')
    parser.add_argument('--TEST_MODE', action='store_true', 
                        help='turning on the testmode means the code will run on a small dataset.')    
    parser.add_argument('-q', '--QUANTILE', default=0.005, type=float,
                        help='the threshold used for activation')
    parser.add_argument('--SEG_THRESHOLD', default=0.04, type=float,
                        help='the threshold used for visualization')
    parser.add_argument('--SCORE_THRESHOLD', default=0.04, type=float,
                        help='the threshold used for IoU score (in HTML file)')
    parser.add_argument('--TOPN', default=10, type=int,
                        help='to show top N image with highest activation for each unit')
    parser.add_argument('--MODEL_PARALLEL', action='store_true', 
                        help='some model is trained in multi-GPU, so there is another way to load them.')        
    parser.add_argument('--BATCH_SIZE', default=128, type=int,
                        help='batch size used in feature extraction')
    parser.add_argument('--WORKERS', default=12, type=int,
                        help='how many workers are fetching images')
    parser.add_argument('--TALLY_BATCH_SIZE', default=16, type=int,
                        help='batch size used in tallying')
    parser.add_argument('--TALLY_AHEAD', default=4, type=int,
                        help='batch size used in tallying')
    parser.add_argument('--INDEX_FILE', default='index.csv', type=str, 
                        help='if you turn on the TEST_MODE, actually you should provide this file on your own')
    return parser

args = get_parser().parse_args()

# trivial settings
args.GPU = True                     # running on GPU is highly suggested
args.CLEAN = True                   # set to "True" if you want to clean the temporary large files after generating result
args.PARALLEL = 1                   # how many process is used for tallying (Experiments show that 1 is the fastest)
args.CATAGORIES = ["object","part","scene","texture","color"] # concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color"

# we will not use broden1_227
args.DATA_DIRECTORY = 'dataset/broden1_224'
args.IMG_SIZE = 224

if args.DATASET == 'places365':
    args.NUM_CLASSES = 365
elif args.DATASET == 'imagenet':
    args.NUM_CLASSES = 1000

args.FEATURE_NAMES = ['layer4']     # the array of layer where features will be extracted

#print(args)

#MODEL = args.arch                           # model arch: resnet18, alexnet, resnet50, densenet161
#OUTPUT_FOLDER = "result/pytorch_"+DATASET+"_"+MODEL # result will be stored in this folder

fo = FeatureOperator(args)

# load model
checkpoint = torch.load(args.MODEL_FILE)
model = torchvision.models.__dict__[checkpoint["arch"]](num_classes=args.NUM_CLASSES)
# DO print(checkpoint['state_dict']) 
state_dict = {k[2:]: v for k,v in checkpoint['state_dict'].items()}
#state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()} 
model.load_state_dict(state_dict)
# setup a hook to extract unit info
for name in args.FEATURE_NAMES:
    model._modules.get(name).register_forward_hook(hook_feature)
if args.GPU:
    model.cuda()
model.eval()


############ STEP 1: feature extraction ###############
features, maxfeature = fo.feature_extraction(model=model)

for layer_id,layer in enumerate(args.FEATURE_NAMES):
############ STEP 2: calculating threshold ############
    thresholds = fo.quantile_threshold(features[layer_id],savepath="quantile.npy")

############ STEP 3: calculating IoU scores ###########
    tally_result = fo.tally(features[layer_id],thresholds,savepath="tally.csv")

############ STEP 4: generating results ###############
    generate_html_summary(fo.data, layer, args, 
                          tally_result=tally_result,
                          maxfeature=maxfeature[layer_id],
                          features=features[layer_id],
                          thresholds=thresholds)
    if args.CLEAN:
        clean(args)
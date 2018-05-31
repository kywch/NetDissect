import os

def clean(args):
    filelist = [f for f in os.listdir(args.OUTPUT_FOLDER) if f.endswith('mmap')]
    for f in filelist:
        os.remove(os.path.join(args.OUTPUT_FOLDER, f))

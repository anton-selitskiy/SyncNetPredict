import sys, time, os, pdb, argparse, pickle, subprocess, glob, cv2
import numpy as np
import pandas as pd
from shutil import rmtree

#import scenedetect
#from scenedetect.video_manager import VideoManager
#from scenedetect.scene_manager import SceneManager
#from scenedetect.frame_timecode import FrameTimecode
#from scenedetect.stats_manager import StatsManager
#from scenedetect.detectors import ContentDetector

from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal


import time, pdb, argparse, subprocess, pickle, os, gzip, glob

#sys.path.append('/content/gdrive/My Drive/SyncNet')
#from detectors import S3FD
from SyncNetInstance import *

# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========


class Config:
    def __init__(self, videofile):
        self.data_dir = 'output/'
        self.videofile = videofile
        self.reference = self.videofile.split('/')[-1]
        self.facedet_scale = 0.25
        self.crop_scale = 0.4
        self.min_track = 100
        self.frame_rate = 25
        self.num_failed_det = 25
        self.min_face_size = 100
        self.initial_model = 'data/syncnet_v2.model'
        self.batch_size = 20
        self.vshift = 15

#df = pd.read_csv('/home/cxu-serve/p1/lchen63/trustyAI/train.csv')
df = pd.read_csv('output/train.csv')

fake = df[df['label'] == 1]
real = df[df['label'] == 0]

fake_list = fake['filename'].to_list()
real_list = real['filename'].to_list()

if not os.path.exists(os.path.join('output','pywork')):
    os.makedirs(os.path.join('output','pywork'))

videos = fake_list[80:81]
#videos =  ['a86fb8f61d4e54eb.mp4'] 
for video in videos:
    video_path = video # '/home/cxu-serve/p1/lchen63/trustyAI/videos/'+video 
    print(video_path)
    opt = Config(video_path)
    setattr(opt, 'tmp_dir', os.path.join(opt.data_dir, 'pytmp'))
    setattr(opt, 'work_dir', os.path.join(opt.data_dir, 'pywork'))

    # ========== DELETE EXISTING DIRECTORIES ==========

    if os.path.exists(opt.tmp_dir):
        rmtree(opt.tmp_dir)

    # ========== MAKE NEW DIRECTORIES ==========

    os.makedirs(opt.tmp_dir)

    # ========== Create wav and Crops ==========

    command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (opt.videofile,os.path.join(opt.tmp_dir,'audiot.wav'))) 
    output = subprocess.call(command, shell=True, stdout=None)

    path_cropss = '911f7bd7f62e18d6/' # '/home/cxu-serve/p1/lchen63/trustyAI/videos/crops/'
    flist = glob.glob(os.path.join(path_cropss,'*__0.png')) # glob.glob(os.path.join(path_cropss,opt.reference.split('.')[0],'*__0.png'))
    for f_name in flist:
      im_cr = cv2.imread(f_name)
      im_cr_name = f_name.split('/')[-1]
      im_cr_name = im_cr_name.split('.')[0]+'.jpg'
      print(im_cr_name)
      cv2.imwrite(os.path.join(opt.tmp_dir,im_cr_name), cv2.resize(im_cr, (224,224)))

    audiotmp    = os.path.join(opt.tmp_dir,'audio.wav')
    audiostart  = 0
    audioend    = len(flist)/opt.frame_rate

    command = ("ffmpeg -y -i %s -ss %.3f -to %.3f %s" % (os.path.join(opt.tmp_dir,'audiot.wav'),audiostart,audioend,audiotmp)) 
    output = subprocess.call(command, shell=True, stdout=None)

    # ==================== LOAD MODEL AND FILE LIST ====================

    s = SyncNetInstance();

    s.loadParameters(opt.initial_model);
    print("Model %s loaded." % opt.initial_model);

    #traks = [0]
    #flist = glob.glob(os.path.join('data', '*.avi')) #os.path.join(opt.crop_dir, '0*.avi'))
    #flist.sort()

    # ==================== GET OFFSETS ====================

  
    #for idx in traks:
    offset, conf, dist, fconf = s.evaluate(opt) #, videofile=fname)
    #    dists.append(dist)
    #fconfs.append(np.mean(fconf))

    #print(video_path, np.mean(fconf))

    # ==================== PRINT RESULTS TO FILE ====================

    os.makedirs(os.path.join(opt.work_dir,opt.reference))

    with open(os.path.join(opt.work_dir, opt.reference, 'activesd.pckl'), 'wb') as fil:
        pickle.dump(dist, fil)

    with open(os.path.join(opt.work_dir,opt.reference, 'fconf.pckl'), 'wb') as fil:
        pickle.dump(fconf, fil)  

    print(opt.reference, fconf)
    

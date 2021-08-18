# SyncNet

This repository contains the copy from https://github.com/joonson/syncnet_python with additional automatic pipeline (main.py) 

Run:
git clone https://github.com/anton-selitskiy/SyncNetPredict.git
pip install python_speech_features
cd SyncNetPredict/
!bash download_model.sh 
python main.py


It takes list of files (from train.csv) and iterates over it, taking *__0.png from crops.
By defalt, it takes data from video 911f7bd7f62e18d6.mp4 and crops from 911f7bd7f62e18d6/ THIS SHOULD BE CHANGED
The output (names of videos and mean distance) is saved in output/pywork/


## Publications
 
```
@InProceedings{Chung16a,
  author       = "Chung, J.~S. and Zisserman, A.",
  title        = "Out of time: automated lip sync in the wild",
  booktitle    = "Workshop on Multi-view Lip-reading, ACCV",
  year         = "2016",
}
```

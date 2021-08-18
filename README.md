# SyncNet

This repository contains the copy from https://github.com/joonson/syncnet_python with additional automatic pipeline (main.py) 

Please cite the paper below if you make use of the software. 

## Dependencies
```
pip install -r requirements.txt
```

In addition, `ffmpeg` is required.


## Demo


Full pipeline:
```
main.py - combination of run_pipeline.py and run_syncnet.py (I didn't use run_visualise)
```

Outputs:
```
$DATA_DIR/pycrop/$REFERENCE/*.avi - cropped face tracks
$DATA_DIR/pywork/$REFERENCE/offsets.txt - audio-video offset values
$DATA_DIR/pyavi/$REFERENCE/video_out.avi - output video (as shown below)
```
<p align="center">
  <img src="img/ex1.jpg" width="45%"/>
  <img src="img/ex2.jpg" width="45%"/>
</p>

## Publications
 
```
@InProceedings{Chung16a,
  author       = "Chung, J.~S. and Zisserman, A.",
  title        = "Out of time: automated lip sync in the wild",
  booktitle    = "Workshop on Multi-view Lip-reading, ACCV",
  year         = "2016",
}
```

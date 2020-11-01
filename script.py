import os
import subprocess

list = os.listdir('dataset/fisheye/ldr/')
list.sort()
for folder in list:
    subprocess.call(f'./fisheye2pano/build/Fisheye2Pano_HDR -i ./dataset/fisheye/hdr/{folder}/ -l ./fisheye2pano/left.txt  -r ./fisheye2pano/right.txt -o ./dataset/360hdr/'.split())

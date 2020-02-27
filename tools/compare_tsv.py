#!/usr/bin/env python


import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

def linesEqual(infile, infileRef):
  print("compare each line")
  with open(infile,"r") as tsv_in_file:
    with open(infileRef,"r") as tsv_in_file_ref:
      i=1
      for line in tsv_in_file:
        line_ref = tsv_in_file_ref.readline()
        if line==line_ref:
          print(i,"equal")
        else:
          print(i,"unequal")
          return False
        i = i+1
  return True

#based on https://github.com/peteanderson80/bottom-up-attention/blob/master/tools/read_tsv.py
def parseFile(fname):
    # Verify we can read a tsv
    in_data = {}
    with open(fname, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
          item['image_id'] = int(item['image_id'])
          item['image_h'] = int(item['image_h'])
          item['image_w'] = int(item['image_w'])
          item['num_boxes'] = int(item['num_boxes'])
          for field in ['boxes', 'features']:
            item[field] = np.frombuffer(base64.b64decode(item[field].encode()),dtype=np.float32).reshape((item['num_boxes'],-1))
          in_data[item['image_id']] = item
          break
    #print(in_data)
    return in_data


if __name__ == '__main__':
  infile = sys.argv[1] if len(sys.argv)>1 else "data/mscoco_imgfeat/train2014_d2obj36_batch.tsv"
  infileRef = sys.argv[2] if len(sys.argv)>2 else "data/mscoco_imgfeat/trainval_36_trainval_resnet101_faster_rcnn_genome_36.tsv_head-n_1"
  print(infile,"vs.",infileRef)
  if(linesEqual(infile,infileRef)):
    print("all lines equal")
  else:
    print("lines unequal, checking columns")
    dict1 = parseFile(infile)
    dict2 = parseFile(infileRef)
    for id in dict1:
      print("id:",id)
      for key in FIELDNAMES:
        equal = dict1[id][key]==dict2[id][key]
        if type(equal)==bool and equal or equal.all():
          print(equal,key,dict1[id][key])
        else:
          print(False,key)
          print(dict1[id][key])
          print("vs.")
          print(dict2[id][key])

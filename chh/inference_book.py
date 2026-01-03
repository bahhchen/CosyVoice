# import sys
# # 强制 utf-8 编码 + 每行自动 flush
# sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
# sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

# import argparse
import os
# import re
# import traceback
# from typing import List, Tuple, Union, Dict, Any
import time
import torch
import ebook

from pathlib import Path

def inference(src_txt_path, chapter):
    # 目录方式
    if not os.path.isdir(src_txt_path):
        return
    
    filelist = os.listdir(src_txt_path)
    for i, file_name in enumerate(filelist):
        # print(f"文件名： {i}:{file_name}")
        if i < chapter:
            continue
        txt_filename = os.path.join(src_txt_path, file_name)

      
def main():
    inference()

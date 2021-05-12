# -*- coding: utf-8 -*-
# @Time    : 2020/12/9 15:21
# @Author  : Mingxing Li
# @FileName: generate_test_json.py
# @Software: PyCharm

import json

in_dir = "/braindat/lab/limx/MitoEM2021/CODE/Author/baseline/pytorch_connectomics-master/configs/MitoEM/im_train_rat.json"
out_dir = "/braindat/lab/limx/MitoEM2021/CODE/Author/baseline/pytorch_connectomics-master/configs/MitoEM/im_test_rat.json"

path = "/braindat/lab/limx/MitoEM2021/MitoEM-R/MitoEM-R/im/im{:04d}.png"

with open(in_dir, 'r') as f:
    data = json.load(f)

start_id = 500
end_id = 999

file_path = []
for i in range(start_id, end_id+1):
    file_path.append(path.format(i))

print(len(file_path))

data['image'] = file_path


with open(out_dir, 'w') as f:
    json.dump(data, f)


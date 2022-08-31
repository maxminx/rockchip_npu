'''
Copyright: Guangdong Telpo Technologies
Author: sun.yongcong(maxsunyc@gmail.com)
Date: 2022-07-05 09:26:07
Description: Algorithm API for image&video analysis
'''

import torch
from models.experimental import attempt_load
model = attempt_load(' fire_smog/best.pt', map_location=torch.device('cpu'))
#model = attempt_load('./yolov5s.pt ', map_location=torch.device('cpu'))
m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
#print(m.anchor_grid[0][0][0][0][0])
anchors = []
for i in [0,1,2,]:
    for j in [0,1,2]:
        #print(j)
        print(m.anchor_grid[i][0][j][0][0])
        anchors = anchors+m.anchor_grid[i][0][j][0][0].numpy().tolist()
print(anchors)
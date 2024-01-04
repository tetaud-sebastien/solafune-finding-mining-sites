import os
import shutil

import json
import timm
import torch
from loguru import logger

all_pretrained_models_available = timm.list_models(pretrained=True)
print(len(all_pretrained_models_available))

x  = torch.randn(1, 3, 512, 512)

list_models = []

index = all_pretrained_models_available.index("ese_vovnet39b.ra_in1k")

for m in all_pretrained_models_available[index+1:]:

    models = {}

    try:

        model = timm.create_model(m, num_classes=1, pretrained=True)
        pred = model(x)
        list_models.append(m)
        logger.info(f'model {m} supported')
        logger.info(f'output shape {pred.shape}')
        models = {'models' : list_models}

        with open('models_2.json', 'w') as fp:
            json.dump(models, fp)

    except:
        logger.error(f'model {m} not supported')
    
    models_path = "/home/sebastien/.cache/huggingface/hub"
    model_name = "models--timm--"+m
    try:
        models_path = os.path.join("/home/sebastien/.cache/huggingface/hub", model_name)
        print(models_path)
        shutil.rmtree(models_path)
    except:
        continue


models = {'models' : list_models}

with open('models_2.json', 'w') as fp:
    json.dump(models, fp)






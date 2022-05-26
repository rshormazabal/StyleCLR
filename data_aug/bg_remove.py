# You need to run the Rembg server

# Usage with Anaconda:
# conda create -y -n rembg python=3.9
# conda activate rembg
# pip install rembg
# rembg s &> log/rembg.log &

# Usage with Docker:
# docker run -d --rm -v $HOME/.u2net:/root/.u2net -p 127.0.0.1:5000:5000 --name rembg-server danielgatis/rembg s

import os
import io

from PIL import Image
import requests

MODELS = {
    "u2net", # Default UNet
    "u2netp", # Light-weight UNet
    "u2net_human_seg", # UNet trained on humans
    "u2net_cloth_seg" # UNet trained on clothes
}

def from_file_path(file_path: str, port: int=5000, model: str='u2netp') -> Image:
    with open(os.path.expanduser(file_path), 'rb') as fd:
        file = fd.read()
    return from_stream(file, port, model)

def from_pil(image: Image, port: int=5000, model: str='u2netp') -> Image:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    return from_stream(imgByteArr.getvalue(), port=port)

def from_url(url: str, port: int=5000, model='u2netp') -> Image:
    assert model in MODELS
    res = requests.get(f'http://localhost:{port}/?model={model}&url={url}')

    if res.status_code not in {200, 201}:
        raise Exception(f"Background remover server error. Status code: {res.status_code}")

    return Image.open(io.BytesIO(res.content))
    
def from_stream(file: bytes, port: int=5000, model='u2netp') -> Image:
    assert model in MODELS
    res = requests.post(f'http://localhost:{port}', data={'model': 'asdf'}, files={'file': file})

    if res.status_code not in {200, 201}:
        raise Exception(f"Background remover server error. Status code: {res.status_code}")

    return Image.open(io.BytesIO(res.content))
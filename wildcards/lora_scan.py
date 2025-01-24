import argparse
import struct
import json
import logging
import os
import sys
from pathlib import Path


def read_from_safetensors(inpath):
    with inpath.open('rb') as infile:
        num_bytes = infile.read(8)
        header_bytes = (struct.unpack('<Q', num_bytes))[0]
        header = infile.read(header_bytes).decode('utf-8')
        data = json.loads(header)
        return data['__metadata__']


def read_from_json(inpath):
    with inpath.open() as infile:
        data = json.loads(infile.read())
        return data


# r'weights?\s*(around|of|from|)\s*(\d+(?:\.\d+)?)(?:(?:-|~|to)\s*(\d+(?:\.\d+)?))'
# r'(\d+(?:\.\d+))\s*(?:(?:-|~|to)\s*(\d+(?:\.\d+)))\s*weights?'

readers = {
    'civitai.info': read_from_json, # (model,description)
    'cm-info.json': read_from_json, # ModelDescription
    'json': read_from_json, # description
    'safetensors': read_from_safetensors,
}

base_model_keys = [
    'ss_base_model_version',
    'baseModel',
    'BaseModel',
    'sd version',
]

normalize_map = {
    'sd_1.5': 'sd15',
    'SD 1.5': 'sd15',
    'SD1': 'sd15',
    'Illustrious': 'ilxl',
    'Pony': 'pony',
    'SDXL 1.0': 'sdxl',
    'SDXL': 'sdxl',
}

model_key_map = {
    'ilxl': 'ilxl',
    'ill': 'ilxl',
    'pdxl': 'pony',
    'pony': 'pony',
    'xl': 'sdxl',
    '': 'sd15',
}

description_keys = [
    ('model', 'description'),
    ('ModelDescription',),
    ('description',),
]

def get_base_model(metadata):
    if not isinstance(metadata, list):
        metadata = [metadata]
    for key in base_model_keys:
        for data in metadata:
            if value := data.get(key):
                logging.debug(value)
                if base_model := normalize_map.get(value):
                    return base_model
    return 'unkn'


def get_base_model_from_name(filename):
    filename = filename.lower()
    for key,value in model_key_map.items():
        if key in filename:
            return value
    return 'unkn'


def get_description(metadata) -> str:
    if not isinstance(metadata, list):
        metadata = [metadata]
    for keys in description_keys:
        for data in metadata:
            cur_data = data
            for key in keys:
                cur_data = cur_data.get(key, {})
            if cur_data != {}:
                return cur_data
    return ''


def test_file(target: Path):
    basename = target.with_suffix('')
    result = []
    for suffix, reader in readers.items():
        p = Path(f'{basename}.{suffix}')
        if p.exists():
            result.append(reader(p))
    print(target)
    print('from name:', get_base_model_from_name(target.name), 'from metadata:', get_base_model(result))
    for idx,data in enumerate(result):
        logging.debug(idx, get_base_model(data))
    print(get_description(result))
    for idx,data in enumerate(result):
        logging.debug(idx, get_description(data))



def test(top: Path):
    if top.is_dir():
        for root,dirs,files in os.walk(top):
            for file in filter(lambda x: x.endswith('.safetensors'), files):
                test_file(Path(root, file))
    else:
        test_file(top)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='lora_scan.py',
        description='Scan LORA files and make support files for wildcards'
    )
    parser.add_argument('filename')
    parser.add_argument('--log', type=str, default='WARN')
    args = parser.parse_args()

    log_level = getattr(logging, args.log.upper())
    if not isinstance(log_level, int):
        raise ValueError('Invalid log level: %s' % args.log)
    logging.basicConfig(level=log_level)

    test(Path(args.filename))
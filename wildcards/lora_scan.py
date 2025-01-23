import struct
import json
import sys
from pathlib import Path


def read_from_safetensors(inpath):
    with inpath.open('rb') as infile:
        bytes = infile.read(8)
        header_bytes = (struct.unpack('<Q', bytes))[0]
        header = infile.read(header_bytes).decode('utf-8')
        data = json.loads(header)
        return data['__metadata__']


def read_from_json(inpath):
    with inpath.open() as infile:
        data = json.loads(infile.read())
        return data


readers = {
    'safetensors': read_from_safetensors,
    'civitai.info': read_from_json,
    'cm-info.json': read_from_json,
    'json': read_from_json
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
    'SD1': 'sd15'
}


def get_base_model(metadata):
    if not isinstance(metadata, list):
        metadata = [metadata]
    for key in base_model_keys:
        for data in metadata:
            if key in data:
                if data[key] in normalize_map:
                    return normalize_map[data[key]]
    return 'unkn'


target = Path(sys.argv[1])
basename = target.with_suffix('')
result = []
for suffix, reader in readers.items():
    p = Path(f'{basename}.{suffix}')
    if p.exists():
        result.append(reader(p))
for i in result:
    print(get_base_model(i))
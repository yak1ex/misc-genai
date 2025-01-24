import argparse
import struct
import json
import logging
import os
from pathlib import Path
from typing import TextIO


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
    'civitai.info': read_from_json,  # (model,description)
    'cm-info.json': read_from_json,  # ModelDescription
    'json': read_from_json,  # description
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


def get_metadata_list(target: Path):
    basename = target.with_suffix('')
    result = []
    for suffix, reader in readers.items():
        p = Path(f'{basename}.{suffix}')
        if p.exists():
            result.append(reader(p))
    return result


def get_base_model(metadata: list[dict] | dict):
    if not isinstance(metadata, list):
        metadata = [metadata]
    for key in base_model_keys:
        for data in metadata:
            if value := data.get(key):
                logging.debug(f'key [{key}] -> value [{value}]')
                if base_model := normalize_map.get(value):
                    return base_model
    return 'unkn'


def get_base_model_from_name(target: Path):
    filename = target.name.lower()
    for key, value in model_key_map.items():
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
    metadata_list = get_metadata_list(target)
    print(target)
    print('from name:', get_base_model_from_name(target),
          'from metadata:', get_base_model(metadata_list))
    for idx, data in enumerate(metadata_list):
        logging.debug(f'[{idx}] -> basemodel = {get_base_model(data)}')
    print(get_description(metadata_list))
    for idx, data in enumerate(metadata_list):
        logging.debug(f'[{idx}] -> description = {get_description(data)}')


def test(top: Path):
    if top.is_dir():
        for root, dirs, files in os.walk(top):
            for file in filter(lambda x: x.endswith('.safetensors'), files):
                test_file(Path(root, file))
    else:
        test_file(top)


override_list_header = """\
{% macro override_basemodel(modelname) -%}
{# ordering from low priority to high priority -#}
{%- set result = 'unkn' -%}"""

override_list_footer = """\
{{ result }}
{%- endmacro %}"""


def override_list_file(target: Path, output_stream: TextIO):
    metadata_list = get_metadata_list(target)
    actual_base_model = get_base_model(metadata_list)
    inferred_base_model = get_base_model_from_name(target)
    if actual_base_model != inferred_base_model:
        print(f"{{% set result = '{actual_base_model}' "
              f"if modelname == '{target.name}' else result -%}}",
              file=output_stream)


def filter_tensors(arg: str):
    return arg.endswith('.safetensors')


def override_list(top: Path, output: Path):
    with open(output, 'w') as output_stream:
        print(override_list_header, file=output_stream)
        if top.is_dir():
            for root, dirs, files in os.walk(top):
                for file in filter(filter_tensors, files):
                    override_list_file(Path(root, file), output_stream)
        else:
            override_list_file(top, output_stream)
        print(override_list_footer, file=output_stream)


def yaml_fragment_file(target: Path, result: dict):
    metadata_list = get_metadata_list(target)
    actual_base_model = get_base_model(metadata_list)
    result.setdefault(actual_base_model, []).append(target.stem)


def yaml_fragment(top: Path, output: Path):
    result = {}
    if top.is_dir():
        for root, dirs, files in os.walk(top):
            for file in filter(filter_tensors, files):
                yaml_fragment_file(Path(root, file), result)
    else:
        yaml_fragment_file(top, result)
    with open(output, 'w') as output_stream:
        for basemodel, loras in result.items():
            print(f'{basemodel}:', file=output_stream)
            for lora in loras:
                print(f'  - <lora:{lora}:1>', file=output_stream)  # TODO: title, weight


if __name__ == '__main__':
    def log_level(level: str) -> int:
        level = getattr(logging, level.upper(), None)
        if not isinstance(level, int):
            raise ValueError(f'Invalid log level: {level}')
        return level

    parser = argparse.ArgumentParser(
        prog='lora_scan.py',
        description='Scan LORA files and make support files for wildcards'
    )
    parser.add_argument('filename', type=Path)
    parser.add_argument('--log', type=log_level, default='WARN')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--jinja', type=Path, help='output jinja filename for basemodel overriding against inferrence')
    parser.add_argument('--yaml', type=Path, help='output YAML filename for wildcards fragment')
    args = parser.parse_args()

    logging.basicConfig(level=args.log)

    if args.jinja:
        override_list(args.filename, args.jinja)
    if args.yaml:
        yaml_fragment(args.filename, args.yaml)
    if args.test or (not args.jinja and not args.yaml):
        test(args.filename)

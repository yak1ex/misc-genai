import argparse
import struct
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable, Optional, TextIO, Tuple


Hints = dict[str, dict]


def read_from_safetensors(inpath: Path) -> Optional[dict]:
    with inpath.open('rb') as infile:
        num_bytes = infile.read(8)
        header_bytes = (struct.unpack('<Q', num_bytes))[0]
        header = infile.read(header_bytes).decode('utf-8')
        data = json.loads(header)
        return data.get('__metadata__', None)


def read_from_json(inpath: Path) -> dict:
    with inpath.open() as infile:
        data = json.loads(infile.read())
        return data


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
    'sd15': 'sd15',
    'sdxl': 'sdxl',
    'ilxl': 'ilxl',
    'pony': 'pony',
}

model_key_map = {
    'pdxl': 'pony',
    'pony': 'pony',
    'ilxl': 'ilxl',
    'pxl': 'pony',
    'ill': 'ilxl',
    'ixl': 'ilxl',
    'il': 'ilxl',
    'xl': 'sdxl',
    '': 'sd15',
}

description_keys = [
    ('model', 'description'),
    ('ModelDescription',),
    ('description',),
]

negative_regexp = r'(?<!do not use a )'
antecedent_regexp = r'(?:use it|(?:weight(?: value|)|apply|strength)s?\s*[:：]?\s*(?:around|of|from|is|is between|))\s*'
range_regexp = r'(\d+(?:\.\d+)?)(?:\s*(?:-|~|to)\s*(\d+(?:\.\d+)?)?)?'
subsequent_regexp = r'\s*weights?'
weight_regexps = [
    'placeholder for lora prompt',
    f'{negative_regexp}{antecedent_regexp}{range_regexp}',
    f'{range_regexp}{subsequent_regexp}'
]

title_keys = [
    ('model', 'name'),
    ('ModelName',),
]

keyword_keys = [
    ('activation text',),
    ('trainedWords',),  # list
    ('TrainedWords',),  # list
]

creator_keys = [
    ('creator', 'username'),
]


def get_metadata_list(target: Path, hints: Hints):
    result = [hints.get(target.name, {})]
    basename = target.with_suffix('')
    for suffix, reader in readers.items():
        p = Path(f'{basename}.{suffix}')
        if p.exists():
            if metadata := reader(p):
                result.append(metadata)
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


def get_recursive(target: dict, keys: Iterable):
    for key in keys:
        target = target.get(key, {})
    return target or ''  # not found => {} => False


def get_description(metadata) -> str:
    if not isinstance(metadata, list):
        metadata = [metadata]
    for keys in description_keys:
        for data in metadata:
            if result := get_recursive(data, keys):
                return result
    return ''


def calc_weight(left: str, right: Optional[str] = None) -> float:
    if right is None:
        right = left
    left_value = float(left)
    right_value = float(right)
    if left_value == 0:  # guard
        return 1.0
    if '.' not in right and right_value > 4 and (left_value - int(left_value)) * 10 < right_value:
        right_value = float(f'{int(left_value)}.{right}')
    value = (left_value + right_value) / 2
    return round(value * 100) / 100


def get_weight_from_description(name: str, input_description: str) -> Tuple[float, str]:
    raw_description = re.sub(r'<--|-->', '', input_description)
    cooked_description = re.sub(r'<.*?>', ' ', raw_description)
    weight_regexps[0] = r'(?:<|&lt;)lora:' + re.escape(name) + r':(\d+(?:\.\d+)?)(dummy)?(?:>|&gt;)'
    for regexp in weight_regexps:
        for description in (raw_description, cooked_description):
            if match := re.search(regexp, description, re.IGNORECASE):
                return calc_weight(*match.group(1, 2)), match.group(0)
    return 1, 'N/A'


def get_title(metadata) -> str:
    if not isinstance(metadata, list):
        metadata = [metadata]
    for keys in title_keys:
        for data in metadata:
            if result := get_recursive(data, keys):
                return result
    return ''


def test_file(target: Path, hints: Hints):
    metadata_list = get_metadata_list(target, hints)
    print('[filename]', target)
    print('[title]', get_title(metadata_list))
    print('[weight]', get_weight_from_description(target.stem, get_description(metadata_list)))
    print('[basemodel]', 'from name:', get_base_model_from_name(target),
          'from metadata:', get_base_model(metadata_list))
    for idx, data in enumerate(metadata_list):
        logging.debug(f'[{idx}] -> basemodel = {get_base_model(data)}')
    print('[description]', get_description(metadata_list))
    for idx, data in enumerate(metadata_list):
        logging.debug(f'[{idx}] -> description = {get_description(data)}')


def test(targets: list[Path], hints: Hints):
    for target in targets:
        if target.is_dir():
            for root, dirs, files in os.walk(target):
                for file in filter(lambda x: x.endswith('.safetensors'), files):
                    test_file(Path(root, file), hints)
        else:
            test_file(target, hints)


override_list_header = """\
{% macro override_basemodel(modelname) -%}
{# ordering from low priority to high priority -#}
{%- set result = 'unkn' -%}"""

override_list_footer = """\
{{ result }}
{%- endmacro %}"""


def override_list_file(target: Path, output_stream: TextIO, hints: Hints):
    metadata_list = get_metadata_list(target, hints)
    actual_base_model = get_base_model(metadata_list)
    if actual_base_model == 'unkn':
        logging.warning(f"Can't detect base model of {target.name}")
    inferred_base_model = get_base_model_from_name(target)
    if actual_base_model != inferred_base_model:
        print(f"{{% set result = '{actual_base_model}' "
              f"if modelname == '{target.name}' else result -%}}",
              file=output_stream)


def filter_tensors(arg: str):
    return arg.endswith('.safetensors')


def override_list(targets: list[Path], output: Path, hints: Hints):
    with open(output, 'w') as output_stream:
        print(override_list_header, file=output_stream)
        for target in targets:
            if target.is_dir():
                for root, dirs, files in os.walk(target):
                    for file in filter(filter_tensors, files):
                        override_list_file(Path(root, file), output_stream, hints)
            else:
                override_list_file(target, output_stream, hints)
        print(override_list_footer, file=output_stream)


def yaml_fragment_file(target: Path, result: dict, hints: Hints):
    metadata_list = get_metadata_list(target, hints)
    actual_base_model = get_base_model(metadata_list)
    result.setdefault(actual_base_model, []).append(target)


def yaml_fragment(targets: list[Path], output: Path, hints: Hints):
    result: dict[str, list[Path]] = {}
    for target in targets:
        if target.is_dir():
            for root, dirs, files in os.walk(target):
                for file in filter(filter_tensors, files):
                    yaml_fragment_file(Path(root, file), result, hints)
        else:
            yaml_fragment_file(target, result, hints)
    with open(output, 'w') as output_stream:
        for basemodel, loras in result.items():
            print(f'{basemodel}:', file=output_stream)
            for lora in loras:
                metadata_list = get_metadata_list(lora, hints)
                title = get_title(metadata_list)
                print(f'  - <lora:{lora.stem}:1> # {title}', file=output_stream)  # TODO: weight


if __name__ == '__main__':
    def log_level(level: str) -> int:
        num_level: Optional[Any] = getattr(logging, level.upper(), None)
        if not isinstance(num_level, int):
            raise ValueError(f'Invalid log level: {level}')
        return num_level

    parser = argparse.ArgumentParser(
        prog='lora_scan.py',
        description='Scan LORA files and make support files for wildcards'
    )
    parser.add_argument('target', type=Path, help='target files or directories', nargs='+')
    parser.add_argument('--log', type=log_level, default='WARN')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--jinja', type=Path, help='output jinja filename for basemodel overriding against inferrence')
    parser.add_argument('--yaml', type=Path, help='output YAML filename for wildcards fragment')
    parser.add_argument('--hint', type=Path, help='JSON filename for metadata override')
    args = parser.parse_args()

    logging.basicConfig(level=args.log)
    hints = {}
    if args.hint:
        with open(args.hint) as hint_file:
            hints = json.loads(hint_file.read())

    if args.jinja:
        override_list(args.target, args.jinja, hints)
    if args.yaml:
        yaml_fragment(args.target, args.yaml, hints)
    if args.test or (not args.jinja and not args.yaml):
        test(args.target, hints)

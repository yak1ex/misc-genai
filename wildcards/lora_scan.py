import argparse
import struct
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, TextIO, Tuple


Hints = dict[str, dict]
Metadata = list[dict] | dict
YamlFragmentBasic = dict[str, list[Path]]
YamlFragmentVariant = dict[str, dict[str, dict[str, list[Path]]]]
YamlFragment = Tuple[YamlFragmentBasic, YamlFragmentVariant]


def read_from_safetensors(inpath: Path) -> Optional[dict]:
    with inpath.open('rb') as infile:
        num_bytes = infile.read(8)
        header_bytes = (struct.unpack('<Q', num_bytes))[0]
        header = infile.read(header_bytes).decode('utf-8')
        data = json.loads(header)
        return data.get('__metadata__', None)


def read_from_json(inpath: Path) -> dict:
    with inpath.open(encoding='utf8') as infile:
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

weight_keys = [
    ('preferred weight',),
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


def get_base_model(metadata: Metadata):
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


def get_value(metadata: Metadata, keys_list: Iterable[Iterable]):
    if not isinstance(metadata, list):
        metadata = [metadata]
    for keys in keys_list:
        for data in metadata:
            if result := get_recursive(data, keys):
                return result
    return ''


def get_description(metadata: Metadata) -> str:
    return get_value(metadata, description_keys)


def get_weight_from_metadata(metadata: Metadata) -> str:
    return get_value(metadata, weight_keys)


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


def get_weight(name: str, metadata: Metadata) -> Tuple[float, str]:
    weight = get_weight_from_metadata(metadata)
    if weight and weight != '0':
        return float(weight), 'metadata'
    description = get_description(metadata)
    return get_weight_from_description(name, description)


def get_title(metadata: Metadata) -> str:
    return get_value(metadata, title_keys)


def get_keywords(metadata: Metadata) -> list[str]:
    result: list[str] | str = get_value(metadata, keyword_keys)
    if isinstance(result, list):
        return result
    elif result == '':
        return []
    else:
        return list(map(lambda x: x.strip(), result.split(',')))


def get_creator(metadata: Metadata) -> str:
    return get_value(metadata, creator_keys)


def summary_file(target: Path, output_stream: TextIO, hints: Hints):
    metadata_list = get_metadata_list(target, hints)
    print('[filename]', target, file=output_stream)
    print('[title]', get_title(metadata_list), file=output_stream)
    print('[weight]', get_weight(target.stem, metadata_list), file=output_stream)
    print('[keywords]', ', '.join(get_keywords(metadata_list)), file=output_stream)
    print('[creator]', get_creator(metadata_list), file=output_stream)
    print('[basemodel]', 'from name:', get_base_model_from_name(target),
          'from metadata:', get_base_model(metadata_list), file=output_stream)
    for idx, data in enumerate(metadata_list):
        logging.debug(f'[{idx}] -> basemodel = {get_base_model(data)}')
    print('[description]', get_description(metadata_list), file=output_stream)
    for idx, data in enumerate(metadata_list):
        logging.debug(f'[{idx}] -> description = {get_description(data)}')


def summary(targets: list[Path], output: Optional[Path], hints: Hints):
    if output is None:
        output = Path('-')
    with open(output, 'w', encoding='utf8') if output != Path('-') else sys.stdout as output_stream:
        for target in targets:
            if target.is_dir():
                for root, dirs, files in os.walk(target):
                    for file in filter(lambda x: x.endswith('.safetensors'), files):
                        summary_file(Path(root, file), output_stream, hints)
            else:
                summary_file(target, output_stream, hints)


def dump_file(target: Path, output_stream: TextIO, hints: Hints):
    metadata_list = get_metadata_list(target, hints)
    print(metadata_list, file=output_stream)


def dump(targets: list[Path], output: Path, hints: Hints):
    with open(output, 'w', encoding='utf8') if output != Path('-') else sys.stdout as output_stream:
        for target in targets:
            if target.is_dir():
                for root, dirs, files in os.walk(target):
                    for file in filter(lambda x: x.endswith('.safetensors'), files):
                        dump_file(Path(root, file), output_stream, hints)
            else:
                dump_file(target, output_stream, hints)


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
    with open(output, 'w', encoding='utf8') as output_stream:
        print(override_list_header, file=output_stream)
        for target in targets:
            if target.is_dir():
                for root, dirs, files in os.walk(target):
                    for file in filter(filter_tensors, files):
                        override_list_file(Path(root, file), output_stream, hints)
            else:
                override_list_file(target, output_stream, hints)
        print(override_list_footer, file=output_stream)


pattern_main = r'(illustrious|pdxl|pony|ponyxl|sdxl|pxl|xl|is|il|p6|v?\d+(\.\d+)?[a-z]?|v\d+[a-z]+\d+)'
pattern_at_end = r'[-_. ]' + pattern_main + r'$|\(' + pattern_main + r'\)$'
pattern_at_mid = r'([-_.])' + pattern_main + r'\d*\1'


def get_normalized_name(name: str):
    flag = True
    while flag:
        name, count = re.subn(pattern_at_end, '', name, flags=re.IGNORECASE)
        flag = count > 0
    flag = True
    while flag:
        name, count = re.subn(pattern_at_mid, r'\1', name, flags=re.IGNORECASE)
        flag = count > 0
    name = re.sub(r'^\s*-', '', name)  # hyphen at the beginning
    name = re.sub(r'[?:,[\]{}#&*!|>]', '', name)  # special characters
    name = re.sub(r'([-_ ])\1+', r'\1', name)  # successive characters
    return name.strip()


def yaml_fragment_file(target: Path, result: YamlFragment, want_variant: bool, hints: Hints):
    metadata_list = get_metadata_list(target, hints)
    actual_base_model = get_base_model(metadata_list)
    result[0].setdefault(actual_base_model, []).append(target)
    if want_variant:
        creator = get_creator(metadata_list)
        normalized_name = get_normalized_name(target.stem)
        variant_base = result[1].setdefault(creator, {}).setdefault(normalized_name, {})
        variant_base.setdefault(actual_base_model, []).append(target)


def yaml_fragment_read(targets: list[Path], want_variant: bool, hints: Hints) -> YamlFragment:
    result: YamlFragment = ({}, {})
    for target in targets:
        if target.is_dir():
            for root, dirs, files in os.walk(target):
                for file in filter(filter_tensors, files):
                    yaml_fragment_file(Path(root, file), result, want_variant, hints)
        else:
            yaml_fragment_file(target, result, want_variant, hints)
    return result


def yaml_fragment_basic(data: YamlFragmentBasic, output_stream: TextIO) -> dict[Path, str]:
    cache: dict[Path, str] = {}
    for basemodel, loras in data.items():
        print(f'{basemodel}:', file=output_stream)
        for lora in loras:
            metadata_list = get_metadata_list(lora, hints)
            title = get_title(metadata_list)
            weight, source = get_weight(lora.name, metadata_list)
            keywords = get_keywords(metadata_list)
            if keywords:
                keywords.insert(0, '')
            keywords_str = ', '.join(keywords)
            cache[lora] = f'  - <lora:{lora.stem}:{weight}>{keywords_str} # {title} [[{source}]]'
            print(cache[lora], file=output_stream)
    return cache


def yaml_fragment_variant(data: YamlFragmentVariant, cache: dict[Path, str], want_all: bool, output_stream: TextIO):
    print('# Basemodel Variants', file=output_stream)
    for creator, variant_bases in data.items():
        for variant_base, variants in variant_bases.items():
            if len(variants) > 1:
                print(f'{variant_base}:', file=output_stream)
                for basemodel, loras in variants.items():
                    print(f'  {basemodel}:', file=output_stream)
                    for lora in loras:
                        print(f'    {cache[lora]}', file=output_stream)
    if want_all:
        print('# Rest Basemodel Variants', file=output_stream)
        for creator, variant_bases in data.items():
            show_creator = False
            for variant_base, variants in variant_bases.items():
                if len(variants) <= 1:
                    if not show_creator:
                        print(f'{creator}:', file=output_stream)
                        show_creator = True
                    print(f'  {variant_base}:', file=output_stream)
                    for basemodel, loras in variants.items():
                        print(f'    {basemodel}:', file=output_stream)
                        for lora in loras:
                            print(f'    {cache[lora]}', file=output_stream)


def yaml_fragment(targets: list[Path], output: Path, want_variant: bool, want_all: bool, hints: Hints):
    result = yaml_fragment_read(targets, want_variant, hints)
    with open(output, 'w', encoding='utf8') as output_stream:
        cache = yaml_fragment_basic(result[0], output_stream)
        # if not want_variant, result[1] is blank
        if result[1]:
            yaml_fragment_variant(result[1], cache, want_all, output_stream)


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
    parser.add_argument('--summary', type=Path, help='output summary info, stdout is used if - is specified')
    parser.add_argument('--dump', type=Path, help='output metadata info, stdout is used if - is specified')
    parser.add_argument('--jinja', type=Path, help='output jinja filename for basemodel overriding against inferrence')
    parser.add_argument('--yaml', type=Path, help='output YAML filename for wildcards fragment')
    parser.add_argument('--variant', action='count', default=0,
                        help='Add basemodel variants to YAML file, --yaml required. '
                        'Output even one model variants if specified 2 times.')
    parser.add_argument('--hint', type=Path, help='JSON filename for metadata override')
    args = parser.parse_args()

    if args.variant and not args.yaml:
        print('--variant requires --yaml')
        parser.print_help()
        exit

    logging.basicConfig(level=args.log)
    hints = {}
    if args.hint:
        with open(args.hint, encoding='utf8') as hint_file:
            hints = json.loads(hint_file.read())

    if args.jinja:
        override_list(args.target, args.jinja, hints)
    if args.yaml:
        yaml_fragment(args.target, args.yaml, args.variant > 0, args.variant > 1, hints)
    if args.dump:
        dump(args.target, args.dump, hints)
    if args.summary or (not args.jinja and not args.yaml and not args.dump):
        summary(args.target, args.summary, hints)

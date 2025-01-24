from jinja2 import Template, Environment, FileSystemLoader


env = Environment(loader=FileSystemLoader('.'))
template = env.get_template('mine/test.jinja')

data = {
    "name": "Jane Doe",
    "models": [
        "xxx_pony_v1.safetensors",
        "xxx-xxx-ponyxl-xxx-xxx.safetensors",
        "xxx-pdxl-v2.safetensors",
        "xxx-xxx_xxx_for_Pony.safetensors",
        "Xxx_xxx_PonyXL_xxx_v01.safetensors",
        "XxxxV3_ill.safetensors",
        "xxx_xxx_xxx_illustrious_xxx.safetensors",
        "xxx_SDXL_V5.safetensors",
        "xxx xxx xxx_XL_V1.0.safetensors",
        "xxx-002.safetensors"
    ]
}

rendered = template.render(data)

print(str(rendered))

from pathlib import Path
from dynamicprompts.generators import RandomPromptGenerator,JinjaGenerator
from dynamicprompts.wildcards.wildcard_manager import WildcardManager

wm = WildcardManager(Path("mine"))

print('via its own syntax template')
generator = RandomPromptGenerator(wildcard_manager=wm)
for _ in range(10):
    print(generator.generate("${ym-=!{0|1}}${ym=__common/mutate2/${ym-}/5__}${ym-} __color-${ym}__ __color-${ym}__"))

print('via Jinja2 template')
generator = JinjaGenerator(wildcard_manager=wm)
for _ in range(10):
    print(generator.generate("{{ random_sample('${ym-=!{0|1}}${ym=__common/mutate2/${ym-}/5__}${ym-} __color-${ym}__ __color-${ym}__') }}"))

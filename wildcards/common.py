import yaml
from collections import OrderedDict
from jinja2 import Template, Environment, FileSystemLoader

prob = [ 1, 2, 3, 4, 5, 10, 15, 20, 90 ]

raw_data = {
  "common": {
    "mutate2": {
      **{ str(orig): {
        "0": str(orig),
        **{ str(p): f'{{{100-p}::{orig}|{p}::{1-orig}}}' for p in prob },
        "100": str(1-orig)
      } for orig in (0,1) }
    },
    "mutate3": {
      **{ str(orig): {
        "0": str(orig),
        **{ str(p): f'{{{200-2*p}::{orig}|{p}::{(orig+1)%3}|{p}::{(orig+2)%3}}}' for p in prob },
        "100": f'{{100::{(orig+1)%3}|100::{(orig+2)%3}}}'
      } for orig in (0,1,2) }
    },
    "normalize2": {
      var1: {
        var2: ''.join(sorted(set((var1,var2)))) for var2 in ('0','1')
      } for var1 in ('0','1')
    },
    "normalize3": {
      var1: {
        var2: {
          var3: ''.join(sorted(set((var1,var2,var3)))) for var3 in ('0','1','2')
        } for var2 in ('0','1','2')
      } for var1 in ('0','1','2')
    },
  }
}


env = Environment(loader=FileSystemLoader('./wildcards'))
template = env.get_template('mine/common.yaml.jinja')

data = yaml.dump(raw_data, sort_keys=False).strip()

template.stream(common=data).dump('wildcards/mine/common.yaml')

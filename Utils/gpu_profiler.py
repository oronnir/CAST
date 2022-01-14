"""
Parse output of nvidia-smi into a python dictionary.
This is very basic!
"""
import os
import subprocess
import pprint

from Animator.utils import eprint


def nvidia_smi(should_print=False):
    sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8").split(os.linesep)

    out_dict = {}

    for item in out_list:
        try:
            if item.__contains__(':'):
                key, val = item.split(':')
                key, val = key.strip(), val.strip()
                out_dict[key] = val
        except ValueError:
            pass
        except Exception as e:
            eprint(f'nvidia_smi raised: {e}', e)
    if should_print:
        pprint.pprint(out_dict)
    return out_dict

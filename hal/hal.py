import ray
import math
import matplotlib.pyplot as plt
import subprocess
import gc
import inspect
import torch
import os

from IPython.display import HTML
from IPython.display import display as Idisplay

from base64 import b64encode

import hal as hal
from hal import io_utils

def init():
  ray.init(ignore_reinit_error=True)

def remote(func):
  '''Wraps a function as a Ray Task, to be inspected with hal.show().'''
  @ray.remote(num_gpus=0.5, max_calls=1)
  def ray_func(*args, **kwargs):
    return func(*args, **kwargs)
  return ray_func.remote

def actor(cls):
  '''Wraps a Ray Actor class to expose underlying class API (i.e. no .remote()).''' 
  assert hasattr(cls, '_ray_from_modified_class')
  class Facade:
    def __init__(self):
      self.actor = cls.remote()
      funcs = [attr for attr, v in inspect.getmembers(self.actor) if not (attr.startswith('_'))]
      for f_name in funcs:
        cls_f = getattr(self.actor, f_name)
        def remote_f(*args, **kwargs):
          return cls_f.remote(*args, **kwargs)
        setattr(self, f_name, remote_f)
  return Facade

def show(o):
  '''If a ObjRef, first waits on it; then shows object as appropriate.'''
  if str(type(o)) == "<class 'ray._raylet.ObjectRef'>":  
    o = ray.get(o)
  # TODO: change this to isinstance(o, (hal.audio.Track)) once interaction
  # with autoreload is fixed.
  if str(type(o)) == "<class 'hal.audio.audio.Track'>":
    return _show_track(o)
  if str(type(o)) == "<class 'hal.audio.audio.Audio'>":
    return _show_audio(o)
  if isinstance(o, list):
    return _show_list(o)
  if isinstance(o, dict):
    return _show_dict(o)
  if isinstance(o, str) and o.endswith('.mp4'):
    return _show_mp4(o)
  else:
    raise NotImplementedError('Cannot call .show() on object of type {}'.format(type(o)))

def clear():
  print("Time\t\t\tGPU\t\tUsed\tTotal")
  result = subprocess.getoutput("nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free --format=csv,noheader")
  print(result)
  gc.collect()
  torch.cuda.empty_cache()
  result = subprocess.getoutput("nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free --format=csv,noheader")
  print(result)

def _show_track(t):
  print(t.name)
  t.play_audio()
  t.show_wave()
  t.show_onsets()
  t.show_chromagram()

def _show_audio(a):
  for t in a.tracks:
    hal.hal.show(a.tracks[t])

def _show_list(l):
  """ Shows a list of imges, given their file paths."""
  if not l or len(l) == 0: return
  for p in l:
    if not os.path.isfile(p):
      raise NotImplementedError('"{}" not a valid path.'.format(p))
  
  for i, p in enumerate(l):
    plt.figure(figsize=(10, 8))
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    plt.tight_layout()
    
    img = plt.imread(p)
    plt.imshow(img)
    
    title = io_utils.name_from_path(p)
    plt.title(title)
    
    plt.show()

def _show_dict(d):
  """ Shows a dictionary of images, given a map of title to paths."""
  if not d or len(d.keys()) == 0: return
  for k in d.keys():
    print(k)
    assert isinstance(d[k], list)
    _show_list(d[k])

def _show_mp4(v):
  """ Plays a locally stored small mp4 video file."""
  mp4 = open(v,'rb').read()
  data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  Idisplay(HTML("""
  <video width=400 controls>
        <source src="%s" type="video/mp4">
  </video>
  """ % data_url))

def version():
  return 1
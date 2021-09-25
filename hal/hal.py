import ray
import time

def show(o):
  if str(type(o)) == "<class 'ray._raylet.ObjectRef'>":  
    o = ray.get(o)
  print(o)

def remote(func):
  '''Wraps a function as a Ray Task, to be inspected with hal.show().'''
  @ray.remote
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

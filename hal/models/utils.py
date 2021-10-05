import torch

def diff(path_a, path_b, key_a=None, key_b=None):
  model_a = torch.load(path_a)
  if key_a != None:
    model_a = model_a[key_a]
  model_b = torch.load(path_b)
  if key_b != None:
    model_b = model_b[key_b]

  print("Iterating through a's keys:")
  for k in model_a.keys():
    if k not in model_b.keys():
      print(f'Key "{k}" Not found in b')
  
  print("Iterating through b's keys:")
  for k in model_b.keys():
    if k not in model_a.keys():
      print(f'Key "{k}" Not found in b')
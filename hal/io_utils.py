import os

import urllib.request

def next_dir(root):
  """Creates root if not exists; Creates n+1st numbered directory within it."""
  assert root is not None
  if not os.path.exists(root):
    print(f'"{root}" does not exist - creating.')
    os.makedirs(root)
  existing_dirs = os.listdir(root)
  existing_dirs = list(filter(lambda d: d.isnumeric(), existing_dirs))
  existing_max = max(int(f) for f in existing_dirs) if len(existing_dirs) > 0 else -1
  new_dir = os.path.join(root, str(existing_max + 1))
  os.mkdir(new_dir)
  return new_dir

def name_from_path(path):
  """Returns filename w/o extension from path."""
  basename = os.path.basename(path)
  if '.' in basename:
    basename = basename.split('.')[0]
  return basename

def download_file(url, filepath):
  """Downloads file at given url to given filepath."""
  if not os.path.exists(os.path.dirname(filepath)):
    os.makedirs(os.path.dirname(filepath))
  urllib.request.urlretrieve(url, filepath)


def create_cache_dir(*args):
  """Creates subdirectories (args) inside hal basedir."""
  # TODO: This needs to be more visible and configurable
  base_dir = os.path.join(os.path.expanduser('~'), '.hal')
  cache_dir = os.path.join(base_dir, *args)
  if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
  return cache_dir
  

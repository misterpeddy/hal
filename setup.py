from importlib_metadata import version 
from setuptools import setup, find_packages

def validate_torch_installation():
    '''Validates needed version of PyTorch and CUDA extensions are pre-installed.'''
    cuda_version_str = 'cu111'
    torch_libs = ['torch', 'torchvision']
    for lib in torch_libs:
        lib_version = version(lib)
        if ('+' not in lib_version) or (cuda_version_str != lib_version.split('+')[1]):
            raise ImportError('''
Needed PyTorch related package are not distributed through pypi and thus cannot be installed automatically. 
Run the following to install them before installing hal and/or refer to https://pytorch.org/get-started/locally/ for more options.

pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html''')

def make_all_deps():
    '''Returns complete list of dependencies.'''
    return [
            'numpy<1.21',
            'ray~=1.0.1',
            'librosa~=0.8.1',
            'demucs~=2.0',
            'matplotlib~=3.4',
            'ipython~=7.0',
            'ninja~=1.10'
            ]

validate_torch_installation()

setup(
    name='hal',
    version='0.1.1',
    url='https://github.com/misterpeddy/hal.git',
    author='Pedram Pejman',
    author_email='dev@peddy.ai',
    description='A Framework for controllable, multi-modal, machine learned hallucinations',
    packages=find_packages(),    
    install_requires=make_all_deps(),
)

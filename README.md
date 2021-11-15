# Hal

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/misterpeddy/hal/blob/master/hal_getting_started.ipynb)

Hal is a toolkit for responsible generation of multi-modal machine-learned hallucinations.

* **toolkit** - Not a library, but a collection of carefully picked tools: ([Ray](https://github.com/ray-project/ray) for compute management, [Weights & Biases](https://wandb.ai/site) for artifact and experiment management, [Colab](https://colab.research.google.com/?utm_source=scs-index) both as a development environment and for dependency management (using its prod runtime).
* **responsible** - [read this](#motivation-responsible-ai)
* **generation** - The goal is to help artists and creatives generate media - everything else is a means to this end.
* **multi-modal** - The focus is on cross-modality generation (text->image, audio->video, etc.).
* **machine-learned hallucinations** - Hal is a tool for artistic expression using generative, machine-learned models - while it provides infrastructure for tasks such as text to speech or ASR, it will not actively support these. As a rule of thumb, if there exist generally accepted "goodness" metrics for its generated media (e.g., WER), you're using it for something it's not meant to do.

Currently, I'm focused on the Audio to Video use case - others (Text -> Audio/Video) will naturally follow.

## Motivation: Responsible AI

In the next few years, I suspect we'll continue to see an acceleration of artists and creatives using ML in their creation process. Hal aims to support this trend in 2 ways:

1. Currently, it's fairly difficult for artists (even those with some programming background) to use ML - they need to deal with developer products/tools to set up environments, run commands/scripts they don't understand, and grapple with low-level system dependencies (e.g., CUDA) - *Hal's short term goal is to provide the best UX possible for artists to train and use generative ML models,* abstracting away the technical details of ML systems with a clean and regular interface.

2. As this space matures and we see more monetization of artwork produced by ML, we'll develop more policy/ethics around how credit attribution should work (if you train a model on my data and then sell its generations, how do you determine what you owe me, if anything at all?) - Hal's been designed with lineage tracking of all artifacts (Models, Datasets, Images, Videos) so that once such policies start to emerge, we'll have machinery to enable artists using Hal to do The Right Thing without changing their workflow. Meaning, when you generated your video, you get a log of what Models/Datasets/etc. were used in the transitive closure of upstream processes that generated your video (i.e. what you did and what people whose work you used did).

So Hal's strategy, if you will, is: 
 1) Offer the best UX
 2) Entice artists to use Hal for their work
 3) As the community settles on some codes of ethics around ML-generated art, build those policies into the tool so that people who want to do the right thing can do it at no extra cost

## Principles

1) Everything that may take a long time is async - We use Ray to distribute computations across cores and/or machines, when available. However, Ray is an implementation detail of the framework and thus is never exposed to the end user.
2) Everything that is returned by any public Hal API must be inspectable with `hal.show()` - this foundational decision allows us to completely decouple the generation of an artifact (which may be happening async) from the materialization of the artifact to disk (which may be remote) from the consumption of the artifact by the end user (since they'll never access artifacts directly, but rather via our inspection API, `hal.show()`)
3) Colab is the only programming environment supported by Hal - Most libraries used in generative modeling have intricate (and sometimes poorly tested) inter-dependencies - instead of trying to poorly support N execution environments, we only support (and test against) the current Colab runtime.
4) All user actions that require deep introspection of a model or inference result are done via Weights & Biases - A Hal user must also be a W&B user to accomplish anything sophisticated (model [re-]training, factorization, etc.).
5) Hi fidelity media generation is expensive - all generative types should provide separate APIs for experimentation and export APIs that re-do computation at higher fidelity when the user is ready to export finalized work. 
6) Every object that can be materialized to disk is an Artifact and must appear as part of the artifact lineage graph. 

## User Guide

I'm continually building out Hal's API in my spare time to match the below user guide. For an uptodate working version, check out the colab badge at the top of the readme.

As mentioned above, the hero use case used to build out Hal is "I have a song; I want to use a generative model to create a music video (where the video is perceived to be driven by/reacting to the audio)"

This is done in 3 stages: 1) Exploring visuals, 2) Analyzing audio, 3) Generating media.

### 1. Explore Visuals

```python

# Arrive at a model

## Use pretrained model
from hal.models import Stylegan2
default_sg2 = Stylegan2()
cfg = {'G_msa': 1024, 'n_mpl': 8}
configured_sg2 = Stylegan2(hp=cfg) # no named variants of a model; modify with hp

## Train model on custom dataset
from hal.io.dataset import ImageDataset, VideoDataset
img_dataset = ImageDataset('~/data/images.zip')
vid_dataset = VideoDataset('youtub.com/v/12345')
new_sg2 = my_sg2.train(img_dataset, finetune=False) # LRO
new_sg2 = new_sg2.train(vid_dataset, finetune=True) # LRO
hal.show(new_sg2) # training status

# Make seed images

## Project an existing image
from hal.io import Image
im = Image('~/data/image.jpg')
projection = my_sg2.project(im) # LRO monitored on W&B (link in hal.show) 
hal.show(projection.images[-1])
hal.show(projection.video)
video = projection.video.export() # LRO monitored on W&B
hal.show(video)

## Sample images from latent space
sampled_images = my_sg2.sample(n=10, seeds=[1, 2, 3])
hal.show(sampled_images)

# Understand the latent space

## Sample points around an image
sampled_images = my_sg2.sample(n=10, center=im)

## Understand and label axes
eigenvecs = my_sg2.factorize()
exploration = my_sg2.explore([im], eigenvecs[:5])
hal.show(exploration)

## Change parameters continuously to see an image morph
exploration = my_sg2.explore([im], truncation=[0.5], steps=range(-1, 1, 20))
hal.show(exploration)
```

### 2. Analyze Audio

```python
# Prepare your audio

## Import from filesystem, a URL or a service provider
from hal.io import Audio
full_song = Audio('~/song.mp3', autogen_stems=True) # LRO

## Cut a part of it
song = full_song.cut(start='00:00', end='00:30') # LRO

## Load some stem manually
song.stems.load({'bass': '~/song_bass.mp3'})

## play your audio file, show waveform, chromogram, onsets for song and stems
hal.show(song)
hal.show(song.stems)

# Edit your audio

## mask out certain time intervals of stems
## apply masked, parameterized filters
```

### 3. Generate Media
```python
# Create a random scene
from hal.gen.scenes import RandomScene
random_sc = RandomScene(my_sg2)
hal.show(r_sc)

## Create a scene from an expliooration
from hal.gen.scenes import ExplorationScene
explore_sc = ExplorationScene(exploration)
hal.show(z)

## Create a scene from an animation
from hal.gen.scenes import AnimationScene
animate_dict = {'truncation': [-0.7, -.7]}
animate_sc = AnimationScene(seed=projection, animate=animate_dict)

## Create an interpolation between two seeds
from hal.gen.scenes import InterpolationScene
interpolate_sc = InterpolationScene(start_seed=projection, 
                                      end_seed=project, strategy='spherical')

## Preview a scene
hal.show(animate_sc)

# Arrange scenes across a timeline
from hal.gen import Timeline
tl = Timeline()
tl.add(random_sc, '00:00')
tl.add(animate_sc, '00:10')
tl.add(interpolate_sc, '00:20')
tl.end('00:30')

# Modulate timeline with music
from hal.audio.analyzers import onsets, chromagram
from hal.audio.mods import noise, morph, bend, brighten

mods = [
        noise(song),
        morph(onsets(song.stems['bass'])),
        bend(chromagram(song.stems['drums'])),
        brighten(onsets(song.stems['vocal'], fmin=200)),
]
tl.modify(mods)

# Preview and export timeline
hal.show(tl)
tl.export('~/data/timeline.mp4')
```

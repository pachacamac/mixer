from pydub import AudioSegment
from pydub.playback import _play_with_ffplay as playback
from pydub.generators import WhiteNoise
import numpy as np
import scipy.interpolate as interp
import os, glob, array
from pathlib import Path

def load(path, lib=None, name=None):
  formats = dict(aiff="aac")
  format = path.split('.')[-1]
  if format in formats:
    format = formats[format]
  try:
    snd = AudioSegment.from_file(path, format)
  except Exception as err:
    print(f'could not load {path}')
    return None
  if lib is not None:
    if name is None:
      name = os.path.split(path)[-1].split('.', 1)[0]
    lib[name] = snd
  return snd

def fetch(url, name, path='.'):
  matches = glob.glob(f"{os.path.join(path, name)}.*")
  if matches:
    return matches[0]
  else:
    os.system(f'youtube-dl -f 0 "{url}" --output "{os.path.join(path, name)}.%(ext)s"')
    return glob.glob(f"{os.path.join(path, name)}.*")[0]

def fetchAndLoad(url, name, path='.', lib={}):
  path = fetch(url, name, path)
  return load(path, lib, name)

def loadAll(path, lib={}):
  paths = Path(path).glob('**/*.*')
  for path in paths:
    load(str(path), lib)
  return lib

def normalize_duration(n) -> int:
  return int(n*1000) if isinstance(n, float) else n

def silence(duration) -> AudioSegment:
  return AudioSegment.silent(normalize_duration(duration))

def whitenoise(duration) -> AudioSegment:
  WhiteNoise().to_audio_segment(duration=normalize_duration(duration)).apply_gain(-40)

def show(snd: AudioSegment, width=120, skip_print=False) -> str:
  str = ''
  for chan in snd.split_to_mono():
    str += showMono(chan, width, skip_print)
  return str

def showMono(snd: AudioSegment, width=120, skip_print=False) -> str:
  data = snd.get_array_of_samples()
  chunkSize = int(len(data) / width)
  chars = []
  for co in range(0, len(data)-chunkSize, chunkSize):
    s = 0
    for c in range(co, co+chunkSize):
      s += abs(data[c])
    chars.append(s / chunkSize)
  multiplier = pow(max(0.0001, max(*chars)), -1)
  chars = [c * multiplier for c in chars]
  cm = ' ▁▂▃▄▅▆▇█'
  str = '|' + ''.join([cm[min(len(cm)-1, int(c*(len(cm)-1)))] for c in chars]) + '|'
  if not skip_print:
    print(str)
  return str

def play(snd: AudioSegment, start=0, end=None, duration=None):
  start = normalize_duration(start)
  if end is not None:
    sound = snd[start:normalize_duration(end)]
  elif duration is not None:
    sound = snd[start:start+abs(normalize_duration(duration))]
  else:
    sound = snd[start:None]
  playback(sound)

def add(sound: AudioSegment, samples=[], position=0) -> list:
  samples.append(dict(sound=sound, position=normalize_duration(position)))
  samples.sort(key=lambda item: item.get("position"))
  return position + len(sound)

def mix(samples=[], normalize=True) -> AudioSegment:
  samples.sort(key=lambda item: item.get("position"))
  length = samples[-1]['position'] + len(samples[-1]['sound'])
  base = silence(length)
  for sample in samples:
    sound = sample['sound'].normalize() if normalize else sample['sound']
    base = base.overlay(sound, position=sample['position'])
  return base

# Monkey patching AudioSegment
def as_echo(self, delay=300, repeats=3, gain_change=-3) -> AudioSegment:
  base = silence(len(self) + repeats * delay)
  for i in range(0, repeats):
    base = base.overlay(self.apply_gain(i * gain_change), i * delay)
  return base
AudioSegment.echo = as_echo

def as_stereo_pan(self, pandirection=1, steps=24) -> AudioSegment:
  base = silence(len(self))
  chunksize = int(len(self) / steps)
  panstep = (2 / steps) * pandirection
  pan = -1 if pandirection == 1 else 1
  for i in range(0, len(self)-chunksize, chunksize):
    base = base.overlay(self[i:i+chunksize].pan(pan), i)
    pan += panstep
  return base
AudioSegment.stereo_pan = as_stereo_pan

def as_reverse(self) -> AudioSegment:
  channels = self.split_to_mono()
  for c in channels:
    c = c._spawn(array.array('h', list(reversed(c.get_array_of_samples()))))
  return AudioSegment.from_mono_audiosegments(*channels)
AudioSegment.reverse = as_reverse

def as_stretch(self, factor) -> AudioSegment:
  channels = self.split_to_mono()
  for i, chan in enumerate(channels):
    c = np.array(chan.get_array_of_samples())
    c_interp = interp.interp1d(np.arange(c.size), c)
    c_stretched = c_interp(np.linspace(0, c.size-1, int(len(c) * factor) ))
    c = c_stretched.astype(int)
    channels[i] = chan._spawn(array.array('h', c))
  return AudioSegment.from_mono_audiosegments(*channels)
AudioSegment.stretch = as_stretch

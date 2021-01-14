from pydub import AudioSegment
from pydub.playback import _play_with_ffplay as playback
from pydub.generators import WhiteNoise
import numpy as np
import scipy.interpolate as interp
import os, glob, array
from pathlib import Path

class LazySoundLibDict(dict):
  def setPath(self, path):
    self.__path__ = path

  def __getitem__(self, key):
    item = dict.__getitem__(self, key)
    if key == '__path__':
      return item
    if isinstance(item, str):
      print(f'loading {key} ...')
      item = fetchAndLoad(item, key, self.__path__)
      dict.__setitem__(self, key, item)
    return item
  
  def __getattr__(self, key):
    try:
      return self.__getitem__(key)
    except KeyError as k:
      raise AttributeError(k)
  
  def __setattr__(self, key, value):
    dict.__setitem__(self, key, value)
  
  def __delattr__(self, key):
    try:
      del self[key]
    except KeyError as k:
      raise AttributeError(k)


def soundLib(str, path='.'):
  lib = LazySoundLibDict()
  lib.setPath(path)
  for name, url in [s.split(None, 1) for s in str.strip().splitlines()]:
    lib[name] = url
  return lib

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

def silence(duration) -> AudioSegment:
  return AudioSegment.silent(duration)

def whitenoise(duration) -> AudioSegment:
  WhiteNoise().to_audio_segment(duration=duration).apply_gain(-40)

def show(snd: AudioSegment, width=180, skip_print=False) -> str:
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
  if end is not None:
    sound = snd[start:end]
  elif duration is not None:
    sound = snd[start:start+abs(duration)]
  else:
    sound = snd[start:None]
  playback(sound)


class Track(list):
  def __init__(self):
    self.t = 0

  def show(self, width=180, skip_print=False) -> str:
    self.sort(key=lambda item: item.get("position", 0))
    scale = width / self.duration()
    str = ''
    for e in self:
      p = e.get("position", 0)
      d = len(e.get("sound"))
      str += ' '*int(p*scale)+'█'*int(d*scale) + "\n"
    if not skip_print:
      print(str)
    return str

  def duration(self):
    last = self[-1]
    return last['position'] + len(last['sound'])

  def add(self, sound: AudioSegment, position=None, gain=0, loop=False, times=1) -> int:
    if position is None:
      position = self.t
    self.append(dict(
      sound=sound,
      position=position,
      gain=gain,
      loop=loop,
      times=times
    ))
    self.sort(key=lambda item: item.get("position", 0))
    self.t = position + len(sound)
    return self.t

  def mix(self, normalize=False) -> AudioSegment:
    self.sort(key=lambda item: item.get("position", 0))
    length = self[-1]['position'] + len(self[-1]['sound'])
    base = silence(length)
    for sample in self:
      sound = sample['sound'].normalize() if normalize else sample['sound']
      base = base.overlay(sound,
        position=sample.get('position', 0),
        gain_during_overlay=sample.get('gain', 0),
        loop=sample.get('loop', False),
        times=sample.get('times', 1)
      )
    return base

# Monkey patching AudioSegment
def patchAudioSegment():
  AudioSegment.name = ''
  AudioSegment.play = lambda self, start=0, end=None, duration=None: play(self, start, end, duration)
  AudioSegment.show = lambda self, width=120, skip_print=False: show(self, width, skip_print)

  def __echo(self, delay=300, repeats=3, gain_change=-3) -> AudioSegment:
    base = silence(len(self) + repeats * delay)
    for i in range(0, repeats):
      base = base.overlay(self.apply_gain(i * gain_change), i * delay)
    return base
  AudioSegment.echo = __echo

  def __stereo_pan(self, pandirection=1, steps=24) -> AudioSegment:
    base = silence(len(self))
    chunksize = int(len(self) / steps)
    panstep = (2 / steps) * pandirection
    pan = -1 if pandirection == 1 else 1
    for i in range(0, len(self)-chunksize, chunksize):
      base = base.overlay(self[i:i+chunksize].pan(pan), i)
      pan += panstep
    return base
  AudioSegment.stereo_pan = __stereo_pan

  def __reverse(self) -> AudioSegment:
    channels = self.split_to_mono()
    for c in channels:
      c = c._spawn(array.array('h', list(reversed(c.get_array_of_samples()))))
    return AudioSegment.from_mono_audiosegments(*channels)
  AudioSegment.reverse = __reverse

  def __stretch(self, factor) -> AudioSegment:
    channels = self.split_to_mono()
    for i, chan in enumerate(channels):
      c = np.array(chan.get_array_of_samples())
      c_interp = interp.interp1d(np.arange(c.size), c)
      c_stretched = c_interp(np.linspace(0, c.size-1, int(len(c) * factor) ))
      c = c_stretched.astype(int)
      channels[i] = chan._spawn(array.array('h', c))
    return AudioSegment.from_mono_audiosegments(*channels)
  AudioSegment.stretch = __stretch

  # TODO: 
  def __trim_silence(self, threshold=0) -> AudioSegment:
    # channels = self.split_to_mono()
    # lmin=0
    # rmax=len(self)
    # for i, chan in enumerate(channels):
    #   for n in chan.get_array_of_samples():
    pass
  AudioSegment.trim_silence = __trim_silence

patchAudioSegment()

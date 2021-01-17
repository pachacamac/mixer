from pydub import AudioSegment
from pydub.playback import _play_with_ffplay as playback
from pydub.generators import WhiteNoise
import pydub.scipy_effects
import numpy as np
import scipy.interpolate as interp
import os, glob, array, math, random
from pathlib import Path

class LazySoundLibDict(dict):
  def setPath(self, path):
    self.__path__ = path

  def __getitem__(self, key):
    item = dict.__getitem__(self, key)
    if key == '__path__':
      return item
    if isinstance(item, str):
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


def load_sounds(str, path='.'):
  lib = LazySoundLibDict()
  lib.setPath(path)
  for name_url in [s.split(None, 1) for s in str.strip().splitlines()]:
    if not name_url[0][0] == '#':
      lib[name_url[0]] = name_url[1] if len(name_url) > 1 else ''
  return lib

def load(path, lib=None, name=None):
  formats = dict(aiff="aac")
  format = path.split('.')[-1]
  if format in formats:
    format = formats[format]
  try:
    print(f'loading "{name}" ... ', end='')
    snd = AudioSegment.from_file(path, format)
    print(prettyDuration(len(snd)))
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
    os.system(f'youtube-dl -f "bestaudio/best" "{url}" --output "{os.path.join(path, name)}.%(ext)s"')
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

def whitenoise(duration=1000) -> AudioSegment:
  return WhiteNoise().to_audio_segment(duration)

def prettyDuration(ms: int) -> str:
  hours, ms = divmod(ms, 3600000)
  minutes, ms = divmod(ms, 60000)
  seconds, ms = divmod(ms, 1000)
  return '%dh %dm %ds %dms' % (hours, minutes, seconds, ms)

def endless_random_list(arr: list):
  random.seed(hash(tuple(arr)))
  while True:
    a = arr.copy()
    random.shuffle(a)
    while a:
      yield a.pop()

class Track(list):
  def __init__(self):
    self.t = 0

  def show(self, width=120, skip_print=False) -> str:
    scale = width / self.duration()
    str = ''
    for e in self:
      p = e.get("position", 0)
      d = len(e.get("sound"))
      str += ' ' * int(p * scale) + '┿' * math.ceil(d * scale) + "\n"
    if not skip_print:
      print(str)
    return str

  def duration(self):
    empty = dict(position=0, sound=[])
    tail = max(empty, *self, key=lambda e: e.get("position")+len(e.get("sound")))
    return tail.get("position") + len(tail.get("sound"))

  def sprinkle(self, sounds, min_gap: int=0, max_gap: int=0, start: int=0, end: int=None):
    sounds = [sounds] if not isinstance(sounds, list) else sounds
    erl = endless_random_list(sounds)
    min_gap, max_gap = min(min_gap, max_gap), max(min_gap, max_gap)
    end = self.duration() if end is None else end
    positions = []
    while True:
      sound = next(erl)
      gap = min_gap if min_gap == max_gap else random.randrange(min_gap, max_gap)
      if start + gap + len(sound) > end:
        break
      t = self.add(sound, start+gap)
      positions.append([sound, start+gap])
      start = t
    return positions


  def add(self, sound: AudioSegment, position=None, gain=0, loop=False, times=1) -> int:
    if position is None:
      position = self.t
    self.append(dict(sound=sound, position=position, gain=gain, loop=loop, times=times))
    self.t = position + len(sound) * times
    return self.t

  def mix(self, normalize=False) -> AudioSegment:
    base = silence(self.duration())
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
  AudioSegment.duration = lambda self: len(self)
  AudioSegment.prettyDuration = lambda self: prettyDuration(len(self))

  def __play(self, start=0, end=None, duration=None):
    if end is not None:
      sound = self[start:end]
    elif duration is not None:
      sound = self[start:start+abs(duration)]
    else:
      sound = self[start:None]
    print(sound.prettyDuration())
    playback(sound)
  
  AudioSegment.play = __play

  def __show(self, width=120, skip_print=False) -> str:
    str = ''
    for chan in self.split_to_mono():
      str += chan.show_mono(width, skip_print)
    return str
  AudioSegment.show = __show

  def __show_mono(self, width=120, skip_print=False) -> str:
    data = self.get_array_of_samples()
    chunkSize = int(len(data) / width)
    chars = []
    for co in range(0, len(data)-chunkSize, chunkSize):
      s = 0
      for c in range(co, co+chunkSize):
        s += abs(data[c])
      chars.append(s / chunkSize)
    multiplier = pow(max(0.0001, max(*chars)), -1)
    chars = [c * multiplier for c in chars]
    cm = '_▁▂▃▄▅▆▇█'
    str = ''.join([cm[min(len(cm)-1, int(c*(len(cm)-1)))] for c in chars])
    if not skip_print:
      print(str)
    return str
  AudioSegment.show_mono = __show_mono

  def __show_vertical(self, width=120, height=320, skip_print=False) -> str:
    str = ''
    halfwidth = int(width / 2)
    channels = self.split_to_mono()
    left, right = channels[0], channels[1] if len(channels) > 1 else None
    dl = left.get_array_of_samples()
    dr = [0]*len(dl) if right is None else right.get_array_of_samples()
    chunkSize = int(len(dl) / height)
    la, ra = [], []
    for co in range(0, len(dl)-chunkSize, chunkSize):
      sl, sr = 0,0
      for c in range(co, co+chunkSize):
        sl += abs(dl[c])
        sr += abs(dr[c])
      la.append(sl / chunkSize)
      ra.append(sr / chunkSize)
    m = pow(max(0.0001, max(*(la+ra))), -1)
    for l,r in zip(la,ra):
      l,r = int(l*m*halfwidth), int(r*m*halfwidth)
      str +=  ' '*(halfwidth - l) + '╠' + '═'*l + '╬' + '═'*r + '╣' + ' '*(halfwidth - r) + "\n"
    if not skip_print:
      print(str)
    return str
  AudioSegment.show_vertical = __show_vertical

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

  def __ltrim_silence(self, threshold=-50.0, chunk_size=10) -> AudioSegment:
    """threshold in dB, chunk_size in ms"""
    trim_ms = 0 # ms
    assert chunk_size > 0 # to avoid infinite loop
    while self[trim_ms:trim_ms+chunk_size].dBFS < threshold and trim_ms < len(self):
      trim_ms += chunk_size
    return self[trim_ms:]
  AudioSegment.ltrim_silence = __ltrim_silence

  def __rtrim_silence(self, threshold=-50.0, chunk_size=10) -> AudioSegment:
    """silence_threshold in dB, chunk_size in ms"""
    trim_ms = 0 # ms
    assert chunk_size > 0 # to avoid infinite loop
    while self[-(trim_ms+chunk_size):-trim_ms].dBFS < threshold and trim_ms < len(self):
      trim_ms += chunk_size
    return self[:-trim_ms]
  AudioSegment.rtrim_silence = __rtrim_silence
  
  def __trim_silence(self, threshold=-50.0, chunk_size=10) -> AudioSegment:
    return self.ltrim_silence(threshold, chunk_size).rtrim_silence(threshold, chunk_size)
  AudioSegment.trim_silence = __trim_silence

  def __edgefade(self, duration=None):
    if duration is None:
      duration = max(20, min(3000, int(len(self) * 0.1)))
    duration = max(1, min(duration, int(len(self) / 2)))
    return self.fade_in(duration).fade_out(duration)
  AudioSegment.edgefade = __edgefade

  def __crossfade(self, snd, duration=500):
    duration = max(1, min(duration, int(min(len(self), len(snd)) / 2)))
    base = AudioSegment.silent(len(self) + len(snd) - duration * 2)
    return base.overlay(self.fade_out(duration), 0).overlay(snd.fade_in(duration), len(self) - duration * 1.4)
  AudioSegment.crossfade = __crossfade

  def __random_projections(self, duration, min_chunk_size=None, max_chunk_size=None, fade_duration=500, debug=False):
    random.seed(hash(self) + duration + min_chunk_size + max_chunk_size + fade_duration)
    if min_chunk_size > max_chunk_size:
      min_chunk_size, max_chunk_size = max_chunk_size, min_chunk_size
    min_chunk_size, max_chunk_size = max(1, min(min_chunk_size, len(self)-1)), max(1, min(max_chunk_size, len(self)-1))
    print(min_chunk_size, max_chunk_size, len(self))
    cl = min_chunk_size if min_chunk_size == max_chunk_size else random.randrange(min_chunk_size, max_chunk_size)
    cp = random.randrange(0, len(self) - cl)
    cs = self[cp:cp+cl]
    while True:
      cl = min_chunk_size if min_chunk_size == max_chunk_size else random.randrange(min_chunk_size, max_chunk_size)
      cp = random.randrange(0, len(self) - cl)
      cs = cs.crossfade(self[cp:cp+cl], fade_duration)
      if debug:
        print(prettyDuration(len(cs)), cp, cl)
      if len(cs) > duration:
        return cs[:duration]
  AudioSegment.random_projections = __random_projections

  def __show_fft(self):
    from scipy.fftpack import fft, fftfreq
    import matplotlib.pyplot as plt
    samplerate, data = self.frame_rate, self.set_channels(1).get_array_of_samples()
    total_samples = len(data)
    limit = int((total_samples / 2) - 1)
    fft_abs = abs(fft(data))
    freqs = fftfreq(total_samples, 1 / samplerate)
    plt.plot(freqs[:limit], fft_abs[:limit])
    plt.title('Frequency spectrum')
    plt.xlabel('Frequency in Hz')
    plt.ylabel('Amplitude')
    plt.show()
  AudioSegment.fft = __fft

patchAudioSegment()

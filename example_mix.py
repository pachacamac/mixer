from mixer import Track, soundLib

lib = soundLib("""
can_open  https://freesound.org/people/InspectorJ/sounds/393878/
drinking  https://freesound.org/people/craigglenday/sounds/517173/
""", "samples")

track = Track()
track.add(lib.can_open[:2000].stretch(1.2).stereo_pan())
track.add(lib.drinking.stretch(1.2).stereo_pan(-1), track.t-1000)
print(track.duration())
mixed = track.mix()
mixed.show()
mixed.play()
#mixed.export("example.mp3", format="mp3", tags={'artist': 'The Coders', 'album': 'Example Mix', 'comments': 'Example'}))

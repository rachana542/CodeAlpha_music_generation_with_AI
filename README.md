# CodeAlpha_music_generation_with_AI
# requirements
* pip install torch music21 scikit-learn numpy

# features
* extracts notes and chords from midi files using music21
* encodes sequences of notes for training
* trains a two-layer LSTM on the encoded sequences
* generates new midi compositions from the trained model

# how to run
* add .mid files into midi_songs folder or update the path in the script to match your dataset
* run the script: python music_generator.py
* after training the model generates a new .mid file named  output.mid which will be saved to desktop

# output
* the script generates some number of notes/chods based on learned patterns and saves them to midi file . you can play the file usingany media player that supports midi

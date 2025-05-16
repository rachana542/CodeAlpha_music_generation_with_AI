import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from music21 import converter, instrument, note, chord, stream
from sklearn.preprocessing import LabelEncoder
import glob
import os
from collections import Counter

print("Working directory:", os.getcwd())
print("MIDI files found:", glob.glob("midi_songs/*.mid"))

def get_notes():
    notes = []
    for file in glob.glob("C:/Users/Admin/Desktop/midi_songs/*.mid"):
        print(f"Parsing: {file}")
        midi = converter.parse(file)
        parts = instrument.partitionByInstrument(midi)
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes

        count = 0
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
                count += 1
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                count += 1
        print(f"Notes extracted from {file}: {count}")
    return notes

notes = get_notes()
print("Total notes found:", len(notes))
print("Note distribution:", Counter(notes).most_common(10))
sequence_length = 100
le = LabelEncoder()
notes_encoded = le.fit_transform(notes)
n_vocab = len(set(notes))

network_input = []
network_output = []

for i in range(0, len(notes_encoded) - sequence_length):
    seq_in = notes_encoded[i:i + sequence_length]
    seq_out = notes_encoded[i + sequence_length]
    network_input.append(seq_in)
    network_output.append(seq_out)

n_patterns = len(network_input)
print(f"Sequences prepared: {n_patterns}")

if len(network_output) == 0:
    raise ValueError("network_output is empty. Check the sequence preparation logic.")

inputs = torch.tensor(network_input, dtype=torch.long)
targets = torch.tensor(network_output, dtype=torch.long)

class MusicRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(MusicRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = MusicRNN(vocab_size=n_vocab, embedding_dim=100, hidden_size=256, output_size=n_vocab)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 20
for epoch in range(epochs):
    model.train()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

def sample(preds, temperature=1.0):
    preds = torch.softmax(preds / temperature, dim=-1).squeeze().detach().numpy()
    preds = preds / np.sum(preds)
    return np.random.choice(range(len(preds)), p=preds)

def generate_notes(model, start, n_vocab, le, n_generate=500, temperature=1.0):
    model.eval()
    prediction_output = []
    pattern = start[:]
    for _ in range(n_generate):
        input_seq = torch.tensor([pattern], dtype=torch.long)
        prediction = model(input_seq)
        index = sample(prediction, temperature=temperature)
        result = le.inverse_transform([index])[0]
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:]
    return prediction_output

start = notes_encoded[:sequence_length]
output_notes = generate_notes(model, list(start), n_vocab, le, temperature=0.8)

def create_midi(prediction_output, filename="output.mid"):
    offset = 0
    output_notes = []
    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes_objs = [note.Note(int(n)) for n in notes_in_chord]
            new_chord = chord.Chord(notes_objs)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            output_notes.append(new_note)
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=filename)

create_midi(output_notes, filename="C:/Users/Admin/Desktop/output.mid")
print("Music generated and saved as output.mid")

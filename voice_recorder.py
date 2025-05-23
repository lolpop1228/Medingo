import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

def list_input_devices():
    print("Available audio input devices:")
    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    for i, dev in enumerate(input_devices):
        print(f"{i}: {dev['name']}")
    return input_devices

def select_device(devices):
    while True:
        choice = input("Enter the number of the device you want to use: ")
        if choice.isdigit() and int(choice) in range(len(devices)):
            return devices[int(choice)]
        else:
            print("Invalid choice. Please enter a valid device number.")

def record_audio(device, duration=5, sample_rate=48000, filename='test_output.wav'):
    print(f"\nRecording from '{device['name']}' for {duration} seconds...")
    sd.default.device = device['index']
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                   channels=1, dtype='int16')
    sd.wait()
    write(filename, sample_rate, audio)
    print(f"Recording saved to '{filename}'")

if __name__ == "__main__":
    devices = list_input_devices()
    selected_device = select_device(devices)
    record_audio(selected_device)

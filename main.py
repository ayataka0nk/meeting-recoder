from time import sleep
import soundfile as sf
from modules import (
    mix_audio,
    record_speaker_multi,
    record_microphone_multi,
    record_mix_audio,
    slice_by_seconds,
    slice_by_silence,
    record_audio_limited,
    speechToText,
)
import numpy as np
from multiprocessing import Process, Queue
from dotenv import load_dotenv

load_dotenv()

DURATION = 60


def main():
    global DURATION
    audio = np.empty(DURATION, dtype=np.float32)
    try:
        with open("transcription.txt", "w", encoding="utf-8", buffering=1) as full_text:
            for peace in slice_by_seconds(
                DURATION, slice_by_silence(record_mix_audio())
            ):
                audio = np.append(audio, peace)
                text = speechToText(
                    peace,
                    language="ja",
                    prompt="",
                )
                full_text.write(text + "\n")
                print(text + "\n")
    except KeyboardInterrupt:
        print("Interrupted")
        sf.write(
            "mixedoutput.wav",
            audio,
            16000,
        )


def record_mix_audio_test():
    audio = np.empty(0, dtype=np.float32)
    try:
        with open("transcription.txt", "w") as f:
            for peace in slice_by_seconds(10, slice_by_silence(record_mix_audio())):
                audio = np.append(audio, peace)
                text = speechToText(
                    peace,
                    language="ja",
                    prompt="",
                )
                f.write(text)
                print(text)

    except KeyboardInterrupt:
        print("Interrupted")
        sf.write(
            "mixedoutput.wav",
            audio,
            16000,
        )


def multiprocess_test():
    speakerQueue = Queue()
    micQueue = Queue()
    speakerProcess = Process(target=record_speaker_multi, args=(speakerQueue,))
    micProcess = Process(target=record_microphone_multi, args=(micQueue,))
    speakerProcess.start()
    micProcess.start()
    speakerAudio = np.empty(0, dtype=np.float32)
    micAudio = np.empty(0, dtype=np.float32)
    try:
        while True:
            newSpeakerAudio = speakerQueue.get()
            newMicAudio = micQueue.get()
            speakerAudio = np.append(speakerAudio, newSpeakerAudio)
            micAudio = np.append(micAudio, newMicAudio)
    except KeyboardInterrupt:
        speakerProcess.terminate()
        micProcess.terminate()
        print("Interrupted")
        sf.write("mixed.wav", mix_audio(speakerAudio, micAudio), 16000)
        sf.write(
            "speaker.wav",
            speakerAudio,
            16000,
        )
        sf.write(
            "mic.wav",
            micAudio,
            16000,
        )
    finally:
        speakerProcess.terminate()
        micProcess.terminate()


def text_test():
    with open("hoge.txt", "w", encoding="utf-8", buffering=1) as f:
        for x in range(10):
            print(x)
            f.write(str(x))
            sleep(1)


def record_audio_limited_test():
    audio = record_audio_limited()
    result = speechToText(audio)
    print(result)
    # sf.write("limited.wav", audio, 16000)


if __name__ == "__main__":
    main()

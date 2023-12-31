import argparse
import io
import os
import sys
import ctypes
import subprocess
import time
import speech_recognition as sr
# import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform

import azure.cognitiveservices.speech as speechsdk

import numpy as np
import librosa
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
import speech_recognition
import math

def ConvertPath(path):
    if sys.platform.startswith('win'):
        #    if sys.platform.startswith('darwin'):
        fixed_path = path.replace('/', '\\')
        return fixed_path
    return path

#字串比較演算法
def match_keyword_sim(keyword, matchkeyword):
    keyword_set = set(keyword)
    matchkeyword_set = set(matchkeyword)
    union_set = keyword_set.union(matchkeyword_set)
    
    arrA = [keyword.count(char) for char in union_set]
    arrB = [matchkeyword.count(char) for char in union_set]
    
    num = sum(arrA[i] * arrB[i] for i in range(len(union_set)))
    numA = sum(arrA[i] ** 2 for i in range(len(union_set)))
    numB = sum(arrB[i] ** 2 for i in range(len(union_set)))
    
    cos = num / (math.sqrt(numA) * math.sqrt(numB))
    
    return cos

DO_DENOISE = False  # 啟用消除雜訊
DEFAULT_MODEL = "openai/whisper-small"  # 使用 HuggingFace 上的 model
# USE_DEFAULT_MODEL=True
USE_DEFAULT_MODEL = False  # 設定是否使用 HuggingFace 上的 model, 或是 SAVEPATH 的 model
SAVEPATH = ConvertPath("./whisper-small-tw4")
SEPFORMER_PATH = ConvertPath("pretrained_models/sepformer-wsj02mix")
SEPFORMER_TEMP = ConvertPath("pretrained_models_tmp/sepformer-wsj02mix")

if DO_DENOISE:  # 這有兩種不同的消除雜訊 model, 你可以試看看效果有沒有差異
    # denoise_model = separator.from_hparams(source="speechbrain/noisy-whisper-resucespeech", savedir='pretrained_models/noisy-whisper-rescuespeech', hparams_file="enhance.yaml")
    denoise_model = separator.from_hparams(source="speechbrain/sepformer-dns4-16k-enhancement",
                                           savedir='pretrained_models/sepformer-dns4-16k-enhancement')

GPUdevice = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu")
sep_model = separator.from_hparams(source=SEPFORMER_PATH, savedir=SEPFORMER_TEMP, run_opts={"device": GPUdevice})
# sep_model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')

Vocals = [50, 5000]  # 設定講話的合法頻率範圍, 設成 2000 是因為不想過濾高頻部分.
# SampleRate = 3000
# SampleRate = 6000
# SampleRate = 7500
# SampleRate = 8000
# SampleRate = 9000
# SampleRate = 10000
SampleRate = 16000
# SampleRate = 24000

parser = argparse.ArgumentParser()
parser.add_argument("--key", type=str)
parser.add_argument("--region", type=str)

parser.add_argument("--model", default="small", help="Model to use",
                    choices=["tiny", "base", "small", "medium", "large"])
parser.add_argument("--non_english", action='store_true',
                    help="Don't use the english model.")
parser.add_argument("--energy_threshold", default=1000,
                    help="Energy level for mic to detect.", type=int)
parser.add_argument("--record_timeout", default=2,
                    help="How real time the recording is in seconds.", type=float)
parser.add_argument("--phrase_timeout", default=3,
                    help="How much empty space between recordings before we "
                         "consider it a new line in the transcription.", type=float)
if 'linux' in platform:
    parser.add_argument("--default_microphone", default='pulse',
                        help="Default microphone name for SpeechRecognition. "
                             "Run this with 'list' to view available Microphones.", type=str)
args = parser.parse_args()
last_read_time = None
# The last time a recording was retreived from the queue.
phrase_time = None
# Current raw audio bytes.
last_sample = bytes()
# Thread safe Queue for passing data from the threaded recording callback.
data_queue = Queue()
# We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
recorder = sr.Recognizer()
recorder.energy_threshold = args.energy_threshold
# Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
recorder.dynamic_energy_threshold = False

# Important for linux users.
# Prevents permanent application hang and crash by using the wrong Microphone

if 'linux' in platform:
    mic_name = args.default_microphone
    if not mic_name or mic_name == 'list':
        print("Available microphone devices are: ")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"Microphone with name \"{name}\" found")
    else:
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if mic_name in name:
                source = sr.Microphone(sample_rate=SampleRate, device_index=index)
                break
else:
    source = sr.Microphone(sample_rate=SampleRate)

# Load / Download model
# model = args.model
# if args.model != "large" and not args.non_english:
#    model = model + ".en"
# audio_model = whisper.load_model(model)

"""
if (USE_DEFAULT_MODEL == True):
    processor = WhisperProcessor.from_pretrained(DEFAULT_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(DEFAULT_MODEL)
else:
    processor = WhisperProcessor.from_pretrained(SAVEPATH)
    model = WhisperForConditionalGeneration.from_pretrained(SAVEPATH, local_files_only=True)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="chinese", task="transcribe")
"""

record_timeout = args.record_timeout
phrase_timeout = args.phrase_timeout

temp_file0 = "test1.wav"
temp_file = "test2"
transcription = ['']
transcription2 = ['']
transcription0 = ['']
similarity = 0.0000
go = True
with source:
    recorder.adjust_for_ambient_noise(source)


def record_callback(_, audio: sr.AudioData) -> None:
    """
    Threaded callback function to recieve audio data when recordings finish.
    audio: An AudioData containing the recorded bytes.
    """
    # Grab the raw bytes and push it into the thread safe queue.
    data = audio.get_raw_data()
    data_queue.put(data)


# Create a background thread that will pass us raw audio bytes.
# We could do this manually but SpeechRecognizer provides a nice helper.
recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

# Cue the user that we're ready to go.
# print("Model loaded. Start listening\n")
readyGo = True
while readyGo:
    try:
        readyF = open('ready.txt', 'w', encoding='utf8')
        readyF.write("is ready")
        readyF.close()
        readyGo = False
    except:
        readyF.close()
        readyGo = True

while True:
    try:
        now = datetime.utcnow()
        if not data_queue.empty():
            phrase_complete = False
            # If enough time has passed between recordings, consider the phrase complete.
            # Clear the current working audio buffer to start over with the new data.
            if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                last_sample = bytes()
                phrase_complete = True
            # This is the last time we received new audio data from the queue.
            phrase_time = now

            # Concatenate our current audio data with the latest audio data.
            while not data_queue.empty():
                data = data_queue.get()
                last_sample += data
                if len(last_sample) > source.SAMPLE_RATE * source.SAMPLE_WIDTH * 4:
                    break
            data_queue = Queue()

            # Use AudioData to convert the raw data to wav data.
            audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
            wav_data = io.BytesIO(audio_data.get_wav_data())

            # Write wav data to the temporary file as bytes.
            with open(temp_file0, 'w+b') as f:
                f.write(wav_data.read())
            f.close()

            if len(last_sample) > source.SAMPLE_RATE * source.SAMPLE_WIDTH * 3:
                last_sample = bytes()

            mode = ""
            f1 = open('mode.txt', 'r', encoding='utf8')
            mode = f1.read()
            f1.close()
            if "2" in mode:
                # ----------------------------------------------------------------------------------------------------------------
                est_sources = sep_model.separate_file(path=temp_file0)
                torchaudio.save(temp_file0, est_sources[:, :, 0].detach().cpu(), 8000)
                torchaudio.save(temp_file + 'b.wav', est_sources[:, :, 1].detach().cpu(), 8000)
                # ----------------------------------------------------------------------------------------------------------------

                """
                if DO_DENOISE:
                    est_sources = denoise_model.separate_file(path=temp_file+'.wav')
                    torchaudio.save(temp_file+'.wav', est_sources[:, :, 0].detach().cpu(), SampleRate)

                speech, _ = librosa.load(temp_file+'.wav')
                freq = np.argmax(np.abs(np.fft.rfft(speech))) * SampleRate / speech.size
                # print('1: ', freq)
                if Vocals[0] < freq < Vocals[1]:
                    input_features = processor(speech, sampling_rate=SampleRate, return_tensors="pt").input_features
                    predicted_ids = model.generate(input_features)
                    transcription = processor.batch_decode(predicted_ids)
                    text = processor.batch_decode(predicted_ids, skip_special_tokens = True)

                    # If we detected a pause between recordings, add a new item to our transcripion.
                    # Otherwise edit the existing one.
                    if phrase_complete:
                        transcription.append(text)
                    #    print("append data")
                    else:
                    #    print("clear data")
                        transcription[-1] = text
                """

                if DO_DENOISE:
                    est_sources = denoise_model.separate_file(path=temp_file + 'b.wav')
                    torchaudio.save(temp_file + 'b.wav', est_sources[:, :, 0].detach().cpu(), SampleRate)

                speech, _ = librosa.load(temp_file + 'b.wav')
                freq = np.argmax(np.abs(np.fft.rfft(speech))) * SampleRate / speech.size
                # print('2: ', freq)
                now = datetime.utcnow()
                if Vocals[0] < freq < Vocals[1]:
                    # input_features = processor(speech, sampling_rate=SampleRate, return_tensors="pt").input_features
                    # predicted_ids = model.generate(input_features)
                    # transcription2 = processor.batch_decode(predicted_ids)
                    # text = processor.batch_decode(predicted_ids, skip_special_tokens=True)

                    """
                    print(1)
                    audio2 = speech_recognition.AudioFile("{}b.wav".format(temp_file))
                    print(2)
                    recognizer = speech_recognition.Recognizer()
                    print(3)
                    with audio2 as source:
                        audioData = recognizer.record(source)
                    print(4)
                    result = recognizer.recognize_azure(audioData, key=args.key, language="zh-TW",
                                                        location=args.region)
                    print(5)
                    if result.__len__() > 0:
                        print(result[0])
                        transcription2.append("['" + result[0] + "']")
                    """
                    speech_config = speechsdk.SpeechConfig(subscription=args.key,
                                                           region=args.region)

                    audio_config = speechsdk.AudioConfig(filename="{}b.wav".format(temp_file))
                    audio_config2 = speechsdk.AudioConfig(filename="{}".format(temp_file0))
                    speech_config.speech_recognition_language = "zh-TW"
                    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config,
                                                                   audio_config=audio_config)
                    speech_recognizer2 = speechsdk.SpeechRecognizer(speech_config=speech_config,
                                                                   audio_config=audio_config2)
                    result = speech_recognizer.recognize_once_async().get()
                    result2 = speech_recognizer2.recognize_once_async().get()
                    word1 = result.text
                    word2 = result2.text
                    
                    similarity = match_keyword_sim(word1, word2)
                    #ii = True
                    #while ii:
                    #    try:
                    #        f0 = open('ddd.txt', 'w', encoding='utf8')
                    #        f0.write('!!!!!!! = ' + str(similarity) + 'word1 = ' + word1 + ', word2 = ' + word2)
                    #        f0.close()
                    #        ii = False
                    #    except:
                    #        f0.close()
                    #        ii = True
                    
                    if similarity >= 0.3:
                        go = False
                    else:
                        transcription2.append("['" + word1 + "']")
                        go = True

            else:
                audio2 = speech_recognition.AudioFile("{}".format(temp_file0))
                recognizer = speech_recognition.Recognizer()
                with audio2 as source:
                    audioData = recognizer.record(source)
                result = recognizer.recognize_azure(audioData, key=args.key, language="zh-TW",
                                                    location=args.region)
                if result.__len__() > 0:
                    transcription2.append("['" + result[0] + "']")
                go = True
                """
                est_sources = sep_model.separate_file(path=temp_file0)
                shutil.copyfile(temp_file0, temp_file + 'a.wav')

                speech, _ = librosa.load(temp_file + 'a.wav')
                freq = np.argmax(np.abs(np.fft.rfft(speech))) * SampleRate / speech.size
                # print('2: ', freq)
                if Vocals[0] < freq < Vocals[1]:
                    input_features = processor(speech, sampling_rate=SampleRate, return_tensors="pt").input_features
                    predicted_ids = model.generate(input_features)
                    transcription2 = processor.batch_decode(predicted_ids)
                    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)

                    # If we detected a pause between recordings, add a new item to our transcripion.
                    # Otherwise edit the existing one.
                    if phrase_complete:
                        transcription2.append(text)
                    #   print("append data")
                    else:
                        #    print("clear data")
                        transcription2[-1] = text
                """

            # Clear the console to reprint the updated transcription2.
            # os.system('cls' if os.name=='nt' else 'clear')

            #go = True
            while go:
                try:
                    f2 = open('output.txt', 'w', encoding='utf8')
                    f2.truncate(0)
                    for line in transcription2:
                        f2.write("2: " + str(line))
                        print("2: ", str(line))
                    f2.close()
                    go = False
                except:
                    f2.close()
                    go = True
            transcription2.clear()

            """
            for line in transcription2:
                print("2: ", line)
            """
            #try:
            #    os.remove(temp_file0)
            #except:
            #    {}
            try:
                os.remove(temp_file + '.wav')
            except:
                {}
            #try:
            #    os.remove(temp_file + 'b.wav')
            #except:
            #    {}
            try:
                os.remove(temp_file + 'a.wav')
            except:
                {}
            # Flush stdout.
            # print('', end='', flush=True)

            # Infinite loops are bad for processors, must sleep.
            sleep(0.1)
    except:
        {}    
    # except KeyboardInterrupt:
    #    break

# print("\n\nTranscription:")
# for line in transcription:
#    print(line)

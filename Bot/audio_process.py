import os 

def from_oga_to_wav(filename):
    os.system("ffmpeg -i {0}.oga  {0}.wav".format(filename))


if __name__ == "__main__":
    from_oga_to_wav("393")
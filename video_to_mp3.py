# Convert the sample videos into mp3 file !!

import os
import subprocess

files = os.listdir("Sample Videos")

for file in files:
    file_number = file.split(" [")[0].split(" #")[1]
    file_name = file.split(" ｜ ")[0]
    print(file_number,file_name)
    subprocess.run(["ffmpeg","-i", f"Sample Videos/{file}", f"Sample Audios/{file_number}_{file_name}.mp3"])
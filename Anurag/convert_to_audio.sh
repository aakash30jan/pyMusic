conda install -c conda-forge librosa
conda install -c conda-forge ffmpeg

ffmpeg -i Anurag_1.mp4 -c copy -map 0:a output_audio.mp4


# Does a song's key or danceability affects a song's popularity?

Music has been an integral part of our culture throughout human history. The success of musicians depends heavily on the popularity of their songs. The top 10 worldwide artists in 2019 generated a combined 1 billion dollars in revenue. Our project focuses on finding how a song’s characteristics affect its popularity, focusing on the song’s key and danceability.

We will attempt to calculate both ATE and CATE for each task.

### pre-requirements.txt
-	Python 3.7 or higher
-	Packages from requirements.txt in this repo (the versions in the file are the versions we used. this also might work with other versions).

### song_data.csv and song_info.csv
- Both files are from the Kaggle dataset and can be found  [here](https://www.kaggle.com/edalrami/19000-spotify-songs)
- Contain different features about each song such as artist name, playlist, duration, acoustics, danceability, energy, loudness, "speechiness" (presence of spoken word in track), audio valence, tempo, and liveness

### causal_key.py / causal_dance.py
- In both files, you can find the python code we use to calculate the ATE/CATE for the first research question (with treatment=key) or the second research question (with treatment=dancebility) accordingly. We used bootstrap to create a Confidence interval for each treatment pair. 
- In order to reproduce the result you can simply run the files. as an output, you will get '.npy' files that contain the CIs for each treatment pair in the form of a python dictionary. Make sure that both song_data.csv and song_info.csv will be in the same location as the python file you are running.
- If you want to calculate the ATE- set the parameter  `CATE = False` in the main function. If you want the find the CATE value set `CATE = True`


__For more information about the project see report.pdf file__

[Shoval Zandberg](https://github.com/shoval-z)

[Noa Shmulevich](https://github.com/noashmul)

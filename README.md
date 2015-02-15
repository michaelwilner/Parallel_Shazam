## Pazam
Music identification using a parallelized version of the Shazam algorithm

### About
We all have listened to a song at a concert or party and thought, “Wow, I really like this song! I wonder who sings it...” Well, Shazam Entertainment, Ltd. solved this problem by creating their application which listens to a recording of a song, identifies the song, and returns the information about it to the end user. Shazam uses a music recognition algorithm developed in the year 2000 written by Avery Li-Chun Wang, the co-founder of Shazam. This algorithm recognizes a song from a noisy audio sample, deals with voice codec compression and network dropouts, has few false positives and a high recognition rate, and most importantly it has to perform quickly over an enormous database of songs.

However, Shazam is not perfect. After recording a song via the microphone, there is still several seconds worth of delay where the CPU must process the song before actually being able to identify it against Shazam’s gigantic song database. These precious few seconds may not seem like much, but to a user trying desperately to find out what song is playing at a concert, it can mean failing to identify the song as it limits the amount of attempts he can make using Shazam before the song is over. With the ever-increasing power of GPUs in mobile devices, we believed that we could optimize Shazam using parallel programming and reduce this delay to a level where the human brain cannot perceive the delay inherent to the song processing step. We used CUDA to illustrate the speed increases one can achieve by parallelizing the Shazam algorithm.

When we researched this algorithm, there were no working open source parallelized versions of an audio recognition system that we could find. We implemented a serial version based on the Shazam algorithm, and then created our final parallelized version which ran on GEM, a CUDA cluster in UIUC. Our final parallelized version yielded a result of about a 10x speedup compared to our serial version, which is a very significant result. The code is now released in Github. Feel free to download it and run it on your own NVidia machines.

Michael Wilner
[Ahmed Suhyl](http://ahmedsuhyl.com)
Cody Van Etten



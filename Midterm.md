## Midterm Roadmap

### Project Idea

Imagification : generate an image from a sound.

### What We Did

For now, we have a script that open a .wav file, and save the sound in a buffer. From this buffer we create time slices, associated with a pixel on the result image, and we apply FFT on them. From the result of the FFT we pick the 3 frequencies with the highest amplitudes and it gave us the RGB value of the associated pixel. After computing all the pixels RGB values, we generate a .png image.

### To Do

- [ ] Sliding windows for a better FFT and frequency utilisation

- [ ] Multithreading to optimise the computation of each pixel RGB values

- [ ] filename to use as argument of the script and parsing for the output name
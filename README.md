## Midterm Roadmap

### Project Idea

Imagification : generate an image from a sound.

### How does it work

This script takes a .wav file and does multiple FFTs to determine the color of each pixel of the image.
The colors values for red, green and blue respectivly corresponds to the intensity peak between : 20 and 250 Hz, 250 and 4000 Hz, above 4000 Hz.
The position of the pixel is determined with the choosen sample time.

The correlation of these two choices then generates a .png image using multithreading to be as fast as possible.


### How to use the code

To run the code, simply use the following command :

``` cargo run -- -f "samples/miaou.wav" -s 480```

* -f allows you to specify the path to the file you want to pass in the algorithm
* -s allows you to specify the image size.

#### Limitations

Be careful about these limitations :

* The specified path can be an absolute path or a path relative to the project directory.
* The software currently only accepts .wav files.
* The maximum size is 1024

#### Notes :

* *If no file is specified the program uses the miaou.wav file.*
* *If no size is specified, the program uses the 480 pixel size.*

use hound;
use num::complex::Complex;
use rustfft::FftPlanner;
use std::env;
use image::{ImageBuffer, RgbImage};

const WIDTH: u32 = 480;
const HEIGHT: u32 = 480;


//TODO - Multithreading
//TODO - Sliding window + step :
// mini = 44100*480*480
// reste = Taille - Mini (neg si pas assez, pos si assez)
// step = 44100 + reste/(480*480 -1)
//TODO - file name as arg + output name 

fn main() {
    // Collect args
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        panic!("File name is missing : \nTry cargo run file.txt")
    };
    let file_path = &args[1];

    let mut image: RgbImage = ImageBuffer::new(WIDTH, HEIGHT);

    let mut reader = hound::WavReader::open(file_path).unwrap();

    let spec = reader.spec();
    let lenth = reader.len() as usize;
    println!("{:?}, {:?}", spec, lenth);

    let image_len = WIDTH * HEIGHT;

    let num_samples = lenth / image_len as usize;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(num_samples);

    for pos in 0..image_len {
        // Slice num_samples bytes from the audio
        let mut signal = reader
            .samples::<i16>()
            .take(num_samples)
            .map(|x| Complex::new(x.unwrap() as f32, 0f32))
            .collect::<Vec<_>>();

        // Apply fft in this slice
        fft.process(&mut signal);
        let amplitudes: Vec<f32> = signal.iter().map(|c| c.norm()).collect();

        // Find the 3 frequencies peaks
        let colors: Vec<(i32, f32)> = get_colors_from_max(amplitudes,num_samples);

        *image.get_pixel_mut(pos % HEIGHT, pos / HEIGHT) = image::Rgb(
            colors.iter()
                .map(|(f, _)| (f * 255 / (num_samples as i32 / 2)) as u8)
                .collect::<Vec<u8>>()
                .try_into()
                .unwrap(),
        );
    }

    image.save("output/output.png").unwrap();
}


fn get_colors_from_max(amplitudes: Vec<f32>, num_samples: usize) -> Vec<(i32, f32)>
{
    let mut max = vec![(0, amplitudes[0]), (0, amplitudes[0]), (0, amplitudes[0])];
    for (freq, amp) in amplitudes.iter().take(num_samples / 2).enumerate() {
        if *amp > max[0].1 {
            max[2] = max[1];
            max[1] = max[0];
            max[0] = (freq as i32, *amp);
        } else if *amp > max[1].1 {
            max[2] = max[1];
            max[1] = (freq as i32, *amp);
        } else if *amp > max[2].1 {
            max[2] = (freq as i32, *amp);
        }
    }
    return max;
}
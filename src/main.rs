use hound;
use image::{ImageBuffer, RgbImage};
use num::complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::env;

const WIDTH: usize = 480 / 4;
const HEIGHT: usize = 480 / 4;
const SAMPLE_RATE: usize = 40000;

//TODO - Multithreading

fn main() {
    // Collect args
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        panic!("File name is missing : \nTry cargo run file.txt")
    };
    let file_path = &args[1];

    gen_image(file_path);
}

fn gen_image(file_path: &String) {
    let mut image: RgbImage = ImageBuffer::new(WIDTH as u32, HEIGHT as u32);

    // Get the signal buffer
    let mut reader = hound::WavReader::open(file_path).unwrap();
    let spec = reader.spec();
    let length = reader.len() as usize;
    let signal = reader
        .samples::<i16>()
        .map(|x| Complex::new(x.unwrap() as f32, 0f32))
        .collect::<Vec<_>>();

    println!("{:?}, {:?}", spec, length);

    let image_len = WIDTH * HEIGHT;

    let mut planner = FftPlanner::new();
    let fft: std::sync::Arc<dyn Fft<f32>> = planner.plan_fft_forward(SAMPLE_RATE);

    // Compute the step forr the sliding window
    let diff = length as f64 - SAMPLE_RATE as f64 * WIDTH as f64 * HEIGHT as f64;
    let step =
        (SAMPLE_RATE as f64 + (diff / (WIDTH as f64 * HEIGHT as f64 - 1.0)).floor()) as usize;

    assert!(step * (image_len - 1) + SAMPLE_RATE <= length);

    for pos in 0..image_len {
        println!("pixel: ({}, {})", (pos % HEIGHT), (pos / HEIGHT));
        let amplitudes = fft_sliding_window(&signal, step, pos, &fft);

        // Find the 3 frequencies peaks
        let colors: Vec<f32> = get_colors_from_max(amplitudes);

        let max = colors.iter().cloned().fold(0. / 0., f32::max);

        *image.get_pixel_mut((pos % HEIGHT) as u32, (pos / HEIGHT) as u32) = image::Rgb(
            colors
                .iter()
                .map(|amp| (amp * 255.0 / max) as u8)
                .collect::<Vec<u8>>()
                .try_into()
                .unwrap(),
        );
    }
    let outpath = "output/".to_owned()
        + file_path
            .split("/")
            .last()
            .expect("aled")
            .split(".")
            .next()
            .unwrap()
        + ".png";
    image.save(outpath).unwrap();
}

fn get_colors_from_max(amplitudes: Vec<f32>) -> Vec<f32> {
    let mut max = vec![amplitudes[20], amplitudes[250], amplitudes[4000]];
    for (freq, amp) in amplitudes.iter().take(SAMPLE_RATE / 2).enumerate() {
        match freq {
            20..250 => {
                if *amp > max[0] {
                    max[0] = *amp;
                }
            }
            250..4000 => {
                if *amp > max[1] {
                    max[1] = *amp;
                }
            }
            4000.. => {
                if *amp > max[2] {
                    max[2] = *amp;
                }
            }
            _ => {}
        }
    }
    return max;
}

fn fft_sliding_window(
    signal: &Vec<Complex<f32>>,
    step: usize,
    pos: usize,
    fft: &std::sync::Arc<dyn Fft<f32>>,
) -> Vec<f32> {
    let mut window = signal
        .iter()
        .skip(pos * step)
        .take(SAMPLE_RATE)
        .cloned()
        .collect::<Vec<_>>();

    // Apply fft in this slice
    fft.process(&mut window);
    let amplitudes: Vec<f32> = window.iter().map(|c| c.norm()).collect();

    return amplitudes;
}

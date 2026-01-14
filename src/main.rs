use hound;
use image::{ImageBuffer, RgbImage};
use num::complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::env;

use std::thread;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;

use std::thread::available_parallelism;

const WIDTH: usize = 480 / 4;
const HEIGHT: usize = 480 / 4;
const SAMPLE_RATE: usize = 40000;

//TODO - Multithreading
// Un thread qui collecte les messages et les places dans l'image
// Autant de thread de possible qui font les FFT)



fn main() {

    // Print available threads
    let nb_threads = Arc::new(AtomicUsize::new(available_parallelism().unwrap().get()));

    // Collect args
    let args: Vec<String> = env::args().collect();
    
    // Get file name from args
    if args.len() < 2 {
        panic!("File name is missing : \nTry cargo run file.txt")
    };
    let file_path = args[1].clone();
    
    // -- Common grounds -- //

    // Get the signal buffer
    let mut reader = hound::WavReader::open(args[1].clone()).unwrap();

    let length = reader.len() as usize;
    let signal = reader
        .samples::<i16>()
        .map(|x| Complex::new(x.unwrap() as f32, 0f32))
        .collect::<Vec<_>>();

    // Image
    let image_len = WIDTH * HEIGHT;

    // FFT
    let mut planner = FftPlanner::new();
    let fft: std::sync::Arc<dyn Fft<f32>> = planner.plan_fft_forward(SAMPLE_RATE);

    // Compute the step for the sliding window
    let diff = length as f64 - SAMPLE_RATE as f64 * WIDTH as f64 * HEIGHT as f64;
    let step =
        (SAMPLE_RATE as f64 + (diff / (WIDTH as f64 * HEIGHT as f64 - 1.0)).floor()) as usize;

    assert!(step * (image_len - 1) + SAMPLE_RATE <= length);


    // -- Threading -- //
    let collecting= Arc::new(AtomicBool::new(true));
    let collecting_doe = collecting.clone();

    let (tx, rx) = mpsc::channel();


    // -- Closures -- //
    let image_maker = move || {
        let mut image: RgbImage = ImageBuffer::new(WIDTH as u32, HEIGHT as u32);
        while collecting_doe.load(Ordering::Acquire) {
            let data: (u32, Vec<f32>) = rx.recv().unwrap();
            let max: f32 = data.1.iter().cloned().fold(0. / 0., f32::max);

            // Place the pixel in the image
            *image.get_pixel_mut(data.0 % HEIGHT as u32, data.0 / HEIGHT as u32) = image::Rgb(
                data.1
                    .iter()
                    .map(|amp| (amp * 255.0 / max) as u8)
                    .collect::<Vec<u8>>()
                    .try_into()
                    .unwrap()
            );
        }

        // -- Generate image -- //
        let outpath = "output/".to_owned() + file_path.split("/").last().expect("aled").split(".").next().unwrap() + ".png";
        image.save(outpath).unwrap();
    };


    // Baby threads
    let mut burrow = Vec::new();    

    for n in 0..available_parallelism().unwrap().get() {
        let tx = tx.clone();
        let collecting_bunbuns = collecting.clone();
        let nb_threads_bunbun = nb_threads.clone();

        burrow.push(thread::spawn(move || {
            // -- Go to each pixel of the image -- //
            for pos in n..(image_len as usize/nb_threads_bunbun.load(Ordering::Acquire)*n) {
                let amplitudes = fft_sliding_window(&signal, step, pos, &fft);
                let colors = get_colors_from_max(amplitudes);
                tx.send((pos as u32,colors)).unwrap();
            }
            collecting_bunbuns.store(false, Ordering::Release);

        }));
    }

    // Doe thread
    let doe = thread::spawn(image_maker);
    doe.join().unwrap();
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
    max
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

    amplitudes
}
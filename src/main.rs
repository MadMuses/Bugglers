use hound;
use num::complex::Complex;
use rustfft::FftPlanner;
use std::env;
use image::{ImageBuffer, RgbImage};


use std::thread;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::atomic::AtomicBool;

const WIDTH: u32 = 480;
const HEIGHT: u32 = 480;
const SAMPLE_RATE: u32 = 5;

//TODO - Multithreading
// Un thread qui collecte les messages et les places dans l'image
// Autant de thread de possible qui font les FFT


//TODO - Sliding window + step :
// mini = 44100*480*480
// reste = Taille - Mini (neg si pas assez, pos si assez)
// step = 44100 + reste/(480*480 -1)


fn main() {
    // Collect args
    let args: Vec<String> = env::args().collect();
    
    // Get file name from args
    if args.len() < 2 {
        panic!("File name is missing : \nTry cargo run file.txt")
    };
    let file_path = Arc::new(args[1].clone());
    let file_path_doe = file_path.clone();
    
    // -- Common grounds -- //

    // Image elements
    let mut image: RgbImage = ImageBuffer::new(WIDTH, HEIGHT);
    let image_len = WIDTH * HEIGHT;

    // Audio elements
    let mut reader = hound::WavReader::open(&args[1]).unwrap();
    let num_samples = reader.len() as usize / image_len as usize;

    // -- Threading -- //
    let collecting= Arc::new(AtomicBool::new(true));
    let collecting_arc1 = collecting.clone();
    let collecting_arc2 = collecting.clone();

    let (tx, rx) = mpsc::channel();



    // -- Closures -- //
    let pixel_maker = move || {
        
        // -- Go to each pixel of the image -- //
        for pos in 0..image_len {

            // Slice num_samples bytes from the audio
            let signal = reader
                .samples::<i16>()
                .take(num_samples)
                .map(|x| Complex::new(x.unwrap() as f32, 0f32))
                .collect::<Vec<_>>();

            let colors = do_fft(signal);
            tx.send((pos,colors)).unwrap();
        }
        collecting_arc1.store(false, Ordering::Release);
    };

    let image_maker = move || {
        while collecting_arc2.load(Ordering::Acquire) {
            let data = rx.recv().unwrap();

            // Place the pixel in the image
            *image.get_pixel_mut(data.0 % HEIGHT, data.0 / HEIGHT) = image::Rgb(
                data.1.iter()
                    .map(|&f| (f * 255 / (SAMPLE_RATE as i32 / 2)) as u8)
                    .collect::<Vec<u8>>()
                    .try_into()
                    .unwrap(),
            );
        }

        // -- Generate image -- //
        let outpath = "output/".to_owned() + file_path_doe.split("/").last().expect("aled").split(".").next().unwrap() + ".png";
        image.save(outpath).unwrap();
    };


    // Baby thread
    let baby = thread::spawn(pixel_maker);

    // Doe thread
    let doe = thread::spawn(image_maker);

    baby.join().unwrap();
    doe.join().unwrap();
}

fn do_fft(mut signal: Vec<Complex<f32>>) -> Vec<i32>{

    // FFT elements
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(SAMPLE_RATE as usize);

    // Apply fft in this slice
    fft.process(&mut signal);
    let amplitudes: Vec<f32> = signal.iter().map(|c| c.norm()).collect();

    // Find the 3 frequencies peaks
    get_colors_from_max(amplitudes)
}


fn get_colors_from_max(amplitudes: Vec<f32>) -> Vec<i32>
{
    let mut max = vec![(0, amplitudes[0]), (0, amplitudes[0]), (0, amplitudes[0])];
    for (freq, amp) in amplitudes.iter().take(SAMPLE_RATE as usize/ 2).enumerate() {
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
    return vec![max[0].0,max[1].0,max[2].0];
}
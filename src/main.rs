use hound;
use image::{ImageBuffer, RgbImage, imageops};
use num::complex::Complex;
use rustfft::{Fft, FftPlanner};

use std::{
    thread, env,
    io::{stdout, Write},
    time::{Instant},
    sync::{Arc,Mutex,mpsc},
    sync::atomic::{AtomicUsize,Ordering},
    thread::{available_parallelism}
};

const VERSION: f32 = 1.0;
const FINAL_WIDTH: u32 = 1024;
const FINAL_HEIGHT: u32 = 1024;
const WIDTH: usize = 480;
const HEIGHT: usize = 480;
const SAMPLE_RATE: usize = 40000;

fn gen_ui() -> (String,usize) {

    let nb_threads = Arc::new(AtomicUsize::new(available_parallelism().unwrap().get()));

    // Define Header and footer
    let mut header = format!("╗");
    let mut footer = format!("\n╟");

    // Define lines to print
    let app_name = format!("\n║  Bugglers : V{}",VERSION);
    let threads = format!("\n║  Using {:?} thread to generate a {}x{} image",nb_threads,FINAL_WIDTH,FINAL_HEIGHT);

    // Get UI size (-2 to remove /n in String)
    let ui_elem = vec![app_name,threads];
    let ui_len = ui_elem.iter().map(|s| s.len()).max().unwrap() -2 ;

    // Adapt header and footer
    for _ in 0..ui_len{
        header = "═".to_owned() + &header;
        footer = footer + "─";
    };

    // Send combination of elements
    (ui_elem.into_iter().fold(
        "\n\n╔".to_owned() + &header,
        |mut acc, val| {
            acc.push_str(&val);

            // Adapt each element to match ui size
            for _ in 0..(ui_len + 2 - val.len()){acc.push_str(" ");}
            acc.push_str("  ║");
            acc
        },
    ) + &footer + "╢\n\n",ui_len)
}


fn gen_processing_ui(current: usize,total: usize,ui_len: usize) -> String{
    let clear_line = format!("\r\x1b[1A");
    let mut new_line = format!("║  Processing pixel : {}/{}", current,total);
    for _ in 0..(ui_len + 2 - new_line.len() -1){new_line.push_str(" ");};
    new_line.push_str("  ║\n╚");
    for _ in 0..(new_line.len()-10){new_line.push_str("═");};
    clear_line + &new_line + "╝"
}

fn gen_last_ui(start: Instant,ui_len: usize) -> String{
    let elapsed = start.elapsed().as_secs();
    let clear_line = format!("\r\x1b[1A");
    let mut new_line = format!("║  Elapsed time:  {}m {}s", elapsed / 60, elapsed % 60);
    for _ in 0..(ui_len + 2 - new_line.len() -1){new_line.push_str(" ");};
    new_line.push_str("  ║\n╚");
    for _ in 0..(new_line.len()-10){new_line.push_str("═");};
    clear_line + &new_line + "╝\n\n\n"
}

fn main() {

    // Collect args
    let args: Vec<String> = env::args().collect();

    // Get file name from args
    if args.len() < 2 {
        panic!("File name is missing : \nTry cargo run {:?}","folder/file.txt")
    };
    let file_path = args[1].clone();

    // Generate UI
    let (ui,ui_len) = gen_ui();
    print!("{}",ui);

    // Start program
    threading(file_path,ui_len);
}

fn threading(file_path: String, ui_len: usize) {
    // -- Common grounds -- //

    // Get the signal buffer
    let mut reader = hound::WavReader::open(file_path.clone()).unwrap();

    let length = reader.len() as usize;
    let signal = Arc::new(Mutex::new(
        reader
            .samples::<i16>()
            .map(|x| Complex::new(x.unwrap() as f32, 0f32))
            .collect::<Vec<_>>(),
    ));

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
    let workers = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel();


    // -- Collector closure -- //
    let image_maker = move || {
        let mut image: RgbImage = ImageBuffer::new(WIDTH as u32, HEIGHT as u32);
        let start = Instant::now();

        for i in 0..image_len {
            let data: (u32, Vec<f32>) = rx.recv().unwrap();
            let max: f32 = data.1.iter().cloned().fold(0. / 0., f32::max);

            print!("{}",gen_processing_ui(i,image_len,ui_len));
            stdout().flush().unwrap();
            // Place the pixel in the image
            *image.get_pixel_mut(data.0 % HEIGHT as u32, data.0 / HEIGHT as u32) = image::Rgb(
                data.1
                    .iter()
                    .map(|amp| (amp * 255.0 / max) as u8)
                    .collect::<Vec<u8>>()
                    .try_into()
                    .unwrap(),
            );
        }

        image = imageops::resize(
            &image,
            FINAL_WIDTH,
            FINAL_HEIGHT,
            imageops::FilterType::Lanczos3,
        );

        // -- Generate image -- //
        let outpath = "output/".to_owned()
            + file_path
                .split("/")
                .last()
                .expect("aled")
                .split(".")
                .next()
                .unwrap()
            + "_"
            + WIDTH.to_string().as_str()
            + "x"
            + HEIGHT.to_string().as_str()
            + ".png";

        image.save(outpath).unwrap();

        print!("{}",gen_last_ui(start,ui_len));
    };

    // -- Collector thread -- //
    let collector = thread::spawn(image_maker);

    // -- Baby threads -- //
    // let mut burrow = Vec::new();
    let num_workers = available_parallelism().unwrap().get();
    let mut pos = 0;
    while pos < image_len {
        if workers.load(Ordering::Acquire) < num_workers {
            workers.fetch_add(1, Ordering::AcqRel);
            let w = workers.clone();
            let tx = tx.clone();
            let s = signal.clone();
            let f = fft.clone();
            thread::spawn(move || {
                let amplitudes = fft_sliding_window(&s, pos, step, &f);
                let colors = get_colors_from_max(amplitudes);
                tx.send((pos as u32, colors)).unwrap();
                w.fetch_sub(1, Ordering::AcqRel);
            });

            pos += 1;
        }
    }

    collector.join().unwrap();
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
    signal: &Arc<Mutex<Vec<Complex<f32>>>>,
    pos: usize,
    step: usize,
    fft: &std::sync::Arc<dyn Fft<f32>>,
) -> Vec<f32> {
    let mut window = signal
        .lock()
        .unwrap()
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

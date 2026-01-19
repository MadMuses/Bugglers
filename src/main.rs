use hound;
use image::{ImageBuffer, RgbImage, imageops};
use num::complex::Complex;
use rustfft::{Fft, FftPlanner};

use std::{
    env,
    io::{Write, stdout},
    sync::atomic::{AtomicUsize, Ordering},
    sync::{Arc, Mutex, mpsc},
    thread,
    thread::available_parallelism,
    time::Instant,
};

const VERSION: f32 = 1.0;
const FINAL_WIDTH: u32 = 1024;
const FINAL_HEIGHT: u32 = 1024;
const WIDTH: usize = 480;
const HEIGHT: usize = 480;
const SAMPLE_RATE: usize = 40000;

fn gen_ui() -> String {
    let nb_threads = Arc::new(AtomicUsize::new(available_parallelism().unwrap().get()));

    // Define Header and footer
    let mut header = format!("╗");
    let mut footer = format!("\n╟");

    // Define lines to print
    let app_name = format!("\n║  Bugglers : V{}", VERSION);
    let threads = format!(
        "\n║  Using {:?} thread to generate a {}x{} image",
        nb_threads, FINAL_WIDTH, FINAL_HEIGHT
    );

    // Get UI size (-2 to remove /n in String)
    let ui_elem = vec![app_name, threads];
    let ui_len = ui_elem.iter().map(|s| s.len()).max().unwrap() - 2;

    // Adapt header and footer
    for _ in 0..ui_len {
        header = "═".to_owned() + &header;
        footer = footer + "─";
    }

    // Send combination of elements
    ui_elem
        .into_iter()
        .fold("\n\n╔".to_owned() + &header, |mut acc, val| {
            acc.push_str(&val);

            // Adapt each element to match ui size
            for _ in 0..(ui_len + 2 - val.len()) {
                acc.push_str(" ");
            }
            acc.push_str("  ║");
            acc
        })
        + &footer
        + "╢\n"
}

fn main() {
    // -- Arguments parsing -- //
    let args: Vec<String> = env::args().collect();

    // Get file name from args
    if args.len() < 2 {
        panic!(
            "File name is missing : \nTry cargo run {:?}",
            "folder/file.txt"
        )
    };
    let file_path = args[1].clone();

    print!("{}", gen_ui());

    threading(file_path);
}

fn threading(file_path: String) {
    // -- Initialisation -- //

    // Signal buffer init
    let mut reader = hound::WavReader::open(file_path.clone()).unwrap();
    let length = reader.len() as usize;
    let signal = Arc::new(Mutex::new(
        reader
            .samples::<i16>()
            .map(|x| Complex::new(x.unwrap() as f32, 0f32))
            .collect::<Vec<_>>(),
    ));

    // Image init
    let image_len = WIDTH * HEIGHT;

    // FFT init
    let mut planner = FftPlanner::new();
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft_forward(SAMPLE_RATE);

    // Sliding window init
    let diff = length as f64 - SAMPLE_RATE as f64 * WIDTH as f64 * HEIGHT as f64;
    let step =
        (SAMPLE_RATE as f64 + (diff / (WIDTH as f64 * HEIGHT as f64 - 1.0)).floor()) as usize;

    assert!(step * (image_len - 1) + SAMPLE_RATE <= length);

    // Threads init
    let workers = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel();

    /*
     * Collects computed RGB values for each pixel from a channel and generate the result image
     */
    let image_maker = move || {
        // -- RGB Data Collection -- //
        let mut image: RgbImage = ImageBuffer::new(WIDTH as u32, HEIGHT as u32);
        let start = Instant::now();

        for i in 0..image_len {
            let data: (u32, [u8; 3]) = rx.recv().unwrap();

            print!("\r║  Processing pixel : {}/{}", i, image_len);
            stdout().flush().unwrap();

            // Place the pixel in the image
            *image.get_pixel_mut(data.0 % HEIGHT as u32, data.0 / HEIGHT as u32) =
                image::Rgb(data.1);
        }

        // -- Image generation -- //
        image = imageops::resize(
            &image,
            FINAL_WIDTH,
            FINAL_HEIGHT,
            imageops::FilterType::Lanczos3,
        );

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

        let elapsed = start.elapsed().as_secs();

        println!("\nElapsed time:  {}m {}s", elapsed / 60, elapsed % 60);
    };

    // -- Collector thread -- //
    let collector = thread::spawn(image_maker);

    // -- Computational threads -- //
    let num_workers = available_parallelism().unwrap().get();
    let mut pos = 0;
    while pos < image_len {
        if workers.load(Ordering::Acquire) < num_workers {
            workers.fetch_add(1, Ordering::AcqRel);
            let w = workers.clone();
            let tx = tx.clone();
            let s = signal.clone();
            let f = fft.clone();
            /*
             * Computes the FFT and the RGB value fr each pixel in a thread, then send it the the image maker
             */
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

/// Returns the RGB value of a pixel from a set of frequencies amplitudes.
/// - Low Frequencies maximum of amplitude is used for the red value
/// - Medium frequencies maximum of amplitude is used for the green value
/// - High frequencies maximum of amplitude is used for the blue value
///
/// # Arguments
///  - amplitudes : Vec<f32>, A set of frequencies amplitudes
///
/// # Example
///  
/// ```
/// let colors = get_colors_from_max(amplitudes);
/// ```
fn get_colors_from_max(amplitudes: Vec<f32>) -> [u8; 3] {
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

    let max_value: f32 = max.iter().cloned().fold(0. / 0., f32::max);

    max.iter()
        .map(|amp| (amp * 255.0 / max_value) as u8)
        .collect::<Vec<u8>>()
        .try_into()
        .unwrap()
}

/// Returns the FFT obtained on a slice of 44100 values of the signal. The starting point of the signal slice is define by the step and the pos.
///
/// # Arguments
///  - signal: &Arc<Mutex<Vec<Complex<f32>>>>, The signal to process
///  - pos: usize, The position of the pixel in the final image
///  - step: usize, The step between each starting point of slices
///  - fft: &Arc<dyn Fft<f32>>, The FFT instance used
///
/// # Example
///  
/// ```
/// let s = signal.clone();
/// let f = fft.clone();
/// let colors = fft_sliding_window(&s, pos, step, &f);
/// ```
fn fft_sliding_window(
    signal: &Arc<Mutex<Vec<Complex<f32>>>>,
    pos: usize,
    step: usize,
    fft: &Arc<dyn Fft<f32>>,
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

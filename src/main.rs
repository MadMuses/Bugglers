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
    cmp::max
};

const VERSION: f32 = 1.0;
const FINAL_WIDTH: u32 = 1024;
const FINAL_HEIGHT: u32 = 1024;
const SAMPLE_RATE: usize = 40000;

/// Generate the body of the tips UI displaying tips on how to use the script
///
/// # Arguments
///  - tip_file: <String>, Tips concerning files
///  - tip_size: <String>, Tips concerning size
///
/// # Example
///  
/// ```
/// print!("{}", gen_tips(i, tip_file, tip_size));
/// ```
fn gen_tips(tip_file: String, tip_size: String) -> String {

    // Define Header and footer
    let mut header = format!("╗");
    let mut footer = format!("\n╚");

    let mut ui_elem= vec![];

    if tip_file != "" || tip_size != "" {
        ui_elem.push(format!("\n║  Tips : "));
        if tip_file != "" {ui_elem.push(format!("\n║    {}",tip_file));}
        if tip_size != "" {ui_elem.push(format!("\n║    {}",tip_size));}
        ui_elem.push(format!("\n║  "));
    } else {
        return format!("")
    }

    let ui_len = ui_elem.iter().map(|s| s.len()).max().unwrap() - 2;

    // Adapt header and footer
    for _ in 0..ui_len {
        header = "═".to_owned() + &header;
        footer = footer + "═";
    }

    
    ui_elem
        .into_iter()
        .fold("╔".to_owned() + &header, |mut acc, val| {
            acc.push_str(&val);

            // Adapt each element to match ui size
            for _ in 0..(ui_len + 2 - val.len()) {
                acc.push_str(" ");
            }
            acc.push_str("  ║");
            acc
        })
        + &footer
        + "╝\n\n"
}


/// Generate the header of the terminal UI. Return also the size used by the UI.
///
/// # Arguments
///  - file_path: <String> Path to the file used
///  - width: <usize> Image width
///  - height: <usize> Image height
///  - warning_file: <String> Warning about files
///  - warning_size: <String> Warning about size
///  - err_file: <String> Error about files
///  - err_size: <String> Error about size
///
/// # Example
///  
/// ```
/// let (ui, ui_len) = gen_ui(file_path, width, height,warning_file,warning_size,err_file,err_size);
/// print!("{}", ui);
/// ```
fn gen_ui(file_path: String, width: usize, height: usize, warning_file: String, warning_size: String, err_file: String,err_size: String) -> (String, usize) {
    let nb_threads = Arc::new(AtomicUsize::new(available_parallelism().unwrap().get()));

    // Define Header and footer
    let mut header = format!("╗");
    let mut footer = format!("\n╟");

    // Define lines to print
    let app_name =  format!("\n║  Bugglers v{}", VERSION);
    let threads =   format!("\n║  Using {:?} threads", nb_threads);
    let img_size =  format!("\n║  Image Size : {}x{} pixels", width, height);
    let file =      format!("\n║  File used : {}",file_path);
    let spacer = format!("\n║  ");

    let blank_process = format!("║  Processing pixel : {}/{}  ║",width*height,width*height);

    // Get UI size (-2 to remove /n in String)
    let mut ui_elem = vec![app_name, threads,format!("\n║  "),img_size,file];

    // Add warnings if they happened
    if warning_file != "" || warning_size != "" {
        ui_elem.push(format!("\n║  "));
        ui_elem.push(format!("\n║  Warning(s) :"));
        if warning_file != "" {ui_elem.push(format!("\n║    {}",warning_file));}
        if warning_size != "" {ui_elem.push(format!("\n║    {}",warning_size));}
    }

    // Add errors if they happened
    if err_file != "" || err_size != "" {
        ui_elem.push(format!("\n║  "));
        ui_elem.push(format!("\n║  Error(s) :"));
        if err_file != "" {ui_elem.push(format!("\n║    {}",err_file));}
        if err_size != "" {ui_elem.push(format!("\n║    {}",err_size));}
    }

    // Add last spacer
    ui_elem.push(spacer);

    let ui_len = ui_elem.iter().map(|s| max(s.len(),blank_process.len())).max().unwrap() - 2;

    // Adapt header and footer
    for _ in 0..ui_len {
        header = "═".to_owned() + &header;
        footer = footer + "─";
    }

    // Send combination of elements
    (
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
            + "╢\n\n",
        ui_len,
    )
}

/// Generate the body of the terminal UI displaying the progress of the script.
///
/// # Arguments
///  - current: <usize>, The last generated pixel
///  - total: <usize>, The total number of pixels
///  - ui_len: <usize>, The size of the terminal UI
///
/// # Example
///  
/// ```
/// print!("{}", gen_processing_ui(i, image_len, ui_len));
/// ```
fn gen_processing_ui(current: usize, total: usize, ui_len: usize) -> String {
    let clear_line = format!("\r\x1b[1A");
    let mut new_line = format!("║  Processing pixel : {}/{}", current, total);
    for _ in 0..(ui_len + 2 - new_line.len() - 1) {
        new_line.push_str(" ");
    }
    new_line.push_str("  ║\n╚");
    for _ in 0..(new_line.len() - 10) {
        new_line.push_str("═");
    }
    clear_line + &new_line + "╝"
}

/// Generate the body of the terminal UI displaying the elapsed time after the generation of the image.
///
/// # Arguments
///  - start: <Instant>, The starting time of the script
///  - ui_len: <usize>, The size of the terminal UI
///
/// # Example
///  
/// ```
/// print!("{}", gen_last_ui(start, ui_len));
/// ```
fn gen_last_ui(start: Instant, ui_len: usize) -> String {
    let elapsed = start.elapsed().as_secs();
    let clear_line = format!("\r\x1b[1A");
    let mut new_line = format!("║  Elapsed time:  {}m {}s", elapsed / 60, elapsed % 60);
    for _ in 0..(ui_len + 2 - new_line.len() - 1) {
        new_line.push_str(" ");
    }
    new_line.push_str("  ║\n╚");
    for _ in 0..(new_line.len() - 10) {
        new_line.push_str("═");
    }
    clear_line + &new_line + "╝\n\n\n"
}

/// Create threads to optimize the computation of FFTs and the generation of the image.
///
/// # Arguments
///  - file_path: <String>, The file path of the audio file to open
///  - width, height: <usize>, The resolution of the generated image
///  - ui_len: <usize>, The size of the terminal UI
///
/// # Example
///  
/// ```
/// threading(file_path, width, height, ui_len);
/// ```
fn threading(file_path: String, width: usize, height: usize, ui_len: usize) {
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
    let image_len = width * height;

    // FFT init
    let mut planner = FftPlanner::new();
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft_forward(SAMPLE_RATE);

    // Sliding window init
    let diff = length as f64 - SAMPLE_RATE as f64 * width as f64 * height as f64;
    let step =
        (SAMPLE_RATE as f64 + (diff / (width as f64 * height as f64 - 1.0)).floor()) as usize;

    assert!(step * (image_len - 1) + SAMPLE_RATE <= length);

    // Threads init
    let workers = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel();

    /*
     * Collects computed RGB values for each pixel from a channel and generate the result image
     */
    let image_maker = move || {
        // -- RGB Data Collection -- //
        let mut image: RgbImage = ImageBuffer::new(width as u32, height as u32);
        let start = Instant::now();

        for i in 0..image_len {
            let data: (u32, [u8; 3]) = rx.recv().unwrap();

            print!("{}", gen_processing_ui(i, image_len, ui_len));
            stdout().flush().unwrap();

            // Place the pixel in the image
            *image.get_pixel_mut(data.0 % height as u32, data.0 / height as u32) =
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
            + width.to_string().as_str()
            + "x"
            + height.to_string().as_str()
            + ".png";
        image.save(outpath).unwrap();

        print!("{}", gen_last_ui(start, ui_len));
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
/// - Low Frequencies (20-250Hz) maximum of amplitude is used for the red value
/// - Medium frequencies (250-4000Hz) maximum of amplitude is used for the green value
/// - High frequencies (4000-20000Hz) maximum of amplitude is used for the blue value
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
    // Collect maximum of amplitudes
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

    // Convert max imum of amplitudes to RGB values
    let max_value: f32 = max.iter().cloned().fold(0. / 0., f32::max);
    max.iter()
        .map(|amp| (amp * 255.0 / max_value) as u8)
        .collect::<Vec<u8>>()
        .try_into()
        .unwrap()
}

/// Returns the FFT obtained on a slice of 40000 values of the signal. The starting point of the signal slice is define by the step and the pos.
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
    // Slice the signal
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

fn main() {
    // -- Arguments parsing -- //
    let mut width: usize = 480;
    let mut height: usize = 480;
    let mut file_path = String::from("samples/miaou.wav");

    let mut file = false;
    let mut size = false;
    let mut err_file = format!("");
    let mut err_size = format!("");
    let args: Vec<String> = env::args().skip(1).collect();
    for (i, a) in args.iter().step_by(2).enumerate() {
        match a.as_str() {
            "-f" | "--file" => {
                file_path = args[i * 2 + 1].clone();
                file = true;
            }
            "-s" | "--size" => {
                let result = args[i * 2 + 1].clone().parse::<usize>();

                if result.is_ok() {
                    width = result.unwrap();
                    height = width.clone();
                    size = true;
                } else {
                    err_size = format!("Wrong size value, generic 480x480 used");
                }
            }
            _ => {
                err_file = format!("Wrong argument: {}", a);
            }
        }
    }

    let mut warning_file = format!("");
    let mut warning_size = format!("");
    let mut tip_file = format!("");
    let mut tip_size = format!("");

    if !file {
        warning_file = format!("Default audio sample 'miaou.wav' used");
        tip_file = format!("try 'cargo run -- -f some_file' to choose a specific file");
    }
    if !size {
        warning_size = format!("Default 480x480 size value used");
        tip_size = format!(
            "try 'cargo run -- -s some_size' to choose a specific size, for example '120'"
        );
    }

    // Generate UI
    let (ui, ui_len) = gen_ui(file_path.clone(), width, height,warning_file,warning_size,err_file,err_size);
    print!("{}", ui);

    // Start program
    threading(file_path, width, height, ui_len);

    print!("{}", gen_tips(tip_file,tip_size));
}

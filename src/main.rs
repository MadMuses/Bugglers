use hound;
use num::complex::Complex;
use rustfft::FftPlanner;
use plotters::prelude::*;

fn main() {
    let mut reader = hound::WavReader::open("samples/darude.wav").unwrap();

    let spec = reader.spec();
    let lenth = reader.len() as usize;
    println!("{:?}, {:?}", spec, lenth);

    let image_len = 480 * 480;

    let num_samples = lenth / image_len;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(num_samples);

    let mut pixels: Vec<Vec<i32>> = vec![];
    for _ in 0..image_len {
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
        let mut max: Vec<(i32, f32)> =
            vec![(0, amplitudes[0]), (0, amplitudes[0]), (0, amplitudes[0])];
        for (freq, amp) in amplitudes.iter().take(num_samples / 2).enumerate() {
            if *amp > max[0].1 {
                max[0] = (freq as i32, *amp);
            } else if *amp > max[1].1 {
                max[1] = (freq as i32, *amp);
            } else if *amp > max[2].1 {
                max[2] = (freq as i32, *amp);
            }
        }

        pixels.push(vec![
            max[0].0 * 255 / (num_samples as i32 / 2),
            max[1].0 * 255 / (num_samples as i32 / 2),
            max[2].0 * 255 / (num_samples as i32 / 2),
        ])
    }
    println!("{:?}", pixels.len());
    println!("{:?}", 480*480);

    save_image(&pixels, "output/image.png").unwrap();
}


fn save_image(pixels: &Vec<Vec<i32>>, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new(output_path, (1024, 768)).into_drawing_area();
    root_area.fill(&WHITE)?;
    //let max_value = pixels.iter().flatten().cloned().fold(f32::MIN, i32::max);
    //let min_value = pixels.iter().flatten().cloned().fold(f32::MAX, i32::min);

    // Creating the chart canvas
    let mut chart = ChartBuilder::on(&root_area)
        //.caption("Image", ("sans-serif", 50))
        .build_cartesian_2d(0..480,0..480)?;

    // Draw
    chart.configure_mesh().disable_mesh().draw()?;
    for (pos, colors) in pixels.iter().enumerate() {
        let color = RGBColor(colors[0] as u8,colors[1] as u8,colors[2] as u8);
        
        chart.draw_series(PointSeries::of_element(
            vec![((pos % 480) as i32, (pos / 480) as i32)],
            1,
            &color,
            &|c, s, st| { EmptyElement::at(c) + Circle::new((0,0), s, st.filled()) }
        ))?;
    }
    root_area.present()?;
    Ok(())
}
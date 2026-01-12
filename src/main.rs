use hound;

fn main() {
    let mut reader = hound::WavReader::open("samples/miaou.wav").unwrap();

    let spec = reader.spec();
    println!("{:?}", spec);

    let sqr_sum = reader.samples::<i16>().fold(0.0, |sqr_sum, s| {
        let sample = s.unwrap() as f64;
        // println!("{}", sample);
        sqr_sum + sample * sample
    });
    println!("RMS is {}", (sqr_sum / reader.len() as f64).sqrt());
}

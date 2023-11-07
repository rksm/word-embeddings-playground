use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();
    let inputs = (0..3)
        .map(|_| {
            let mut v = Vec::new();
            for _ in 0..1000000 {
                v.push(rng.gen_range(0.0..1.0));
            }
            v
        })
        .collect::<Vec<_>>();

    let start = std::time::Instant::now();

    for a in &inputs {
        for b in &inputs {
            let c = dot_product(a, b);
            // println!("dot_product({:?}, {:?}) = {}", a, b, c);
            println!("dot_product {c}");

            // let s = cosine_similarity(a, b);
            let s = cosine_similarity_efficiently(a, b);
            // println!("cosine_similarity({:?}, {:?}) = {}", a, b, s);
            println!("cosine_similarity {s}");
        }
    }

    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);
}

#[inline(always)]
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[allow(dead_code)]
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    dot_product(a, b) / (dot_product(a, a) * dot_product(b, b)).sqrt()
}

fn cosine_similarity_efficiently(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut a_squared = 0.0;
    let mut b_squared = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
        a_squared += a[i] * a[i];
        b_squared += b[i] * b[i];
    }
    sum / (a_squared * b_squared).sqrt()
}

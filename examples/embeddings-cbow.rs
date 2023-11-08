use std::path::PathBuf;

use candle_core::Tensor;
use word_embeddings_playground::vocab_builder;
use word_embeddings_playground::word2vec;

fn main() {
    let vocab = vocab_builder::Vocab::from_files().expect("Failed to build vocab");

    if false {
        let words = [
            "steuerberatung",
            "steuersparnissen",
            "steuernachzahlung",
            "steuerbüros",
            "erbschaftsteuern",
            "erbschaftssteuer",
            "einkommenssteuer",
            "einkommensteuerrate",
            "festzuschreiben",
            "steueroptimiert",
            "steuerrückstellung",
            "vermögenssteuerpflichtig",
            "lohnsteuern",
            "ertragssteuern",
            "mineralsteuern",
            "kapitalsteuer",
            "abgeltungssteuer",
            "steuervergünstigung",
        ];

        for word in words {
            let idx = vocab.encode(word).unwrap();
            println!("{word:?} = {idx}");
        }
    }

    if false {
        let words = &vocab.words;

        let embeddings = load_embeddings1();

        for w1 in words {
            for w2 in words {
                let idx1 = vocab.encode(w1).unwrap();
                let idx2 = vocab.encode(w2).unwrap();

                let a = get_embedding_at(&embeddings, idx1).unwrap();
                let b = get_embedding_at(&embeddings, idx2).unwrap();

                let sim = cosine_similarity(&a, &b).unwrap();
                let val = sim.to_scalar::<f32>().unwrap();

                if val > 0.3 && val != 1.0 {
                    println!("similarity: {w1:?} vs {w2:?} = {:?}", val);
                }
            }
        }
    }

    if true {
        let embeddings = load_embeddings3();
        let idx1 = vocab.encode("tech").unwrap();
        let idx2 = vocab.encode("talk").unwrap();
        let a = get_embedding_at(&embeddings, idx1).unwrap();
        let b = get_embedding_at(&embeddings, idx2).unwrap();
        let sim = cosine_similarity(&a, &b).unwrap();
        let val = sim.to_scalar::<f32>().unwrap();
        println!("{val}");
    }
}

#[allow(dead_code)]
pub fn cosine_similarity(a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    dot_product(a, b)? / (dot_product(a, a)? * &dot_product(b, b)?)?.sqrt()?
}

pub fn dot_product(a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    (a * b)?.sum(0)
}

fn get_embedding_at(embeddings: &Tensor, idx: u32) -> candle_core::Result<Tensor> {
    let idx_tensor = Tensor::from_vec(vec![idx], (1,), &word2vec::DEVICE)?;
    let x = embeddings.embedding(&idx_tensor)?;
    x.reshape((100,))
}

fn load_embeddings1() -> Tensor {
    let nn_file = PathBuf::from("data/doppelgaenger/cbow.nn");
    let nn =
        word2vec::Word2VecCbow::load(nn_file, word2vec::EMBEDDING_SIZE).expect("Failed to load");
    let embeddings = nn.projection_layer.embeddings();
    embeddings.clone()
}

fn load_embeddings2() -> Tensor {
    let nn_file = PathBuf::from("data/doppelgaenger/w2v_skipgram.nn");
    let nn = word2vec::Word2VecSkipGram::load(nn_file, word2vec::EMBEDDING_SIZE)
        .expect("Failed to load");
    let embeddings = nn.out_embed.embeddings();
    embeddings.clone()
}

fn load_embeddings3() -> Tensor {
    let nn_file = PathBuf::from("data/doppelgaenger/w2v_skipgram.nn");
    let nn = word2vec::Word2VecSkipGram::load(nn_file, word2vec::EMBEDDING_SIZE)
        .expect("Failed to load");
    let embeddings = nn.in_embed.embeddings();
    embeddings.clone()
}

use anyhow::Result;
use kdam::tqdm;
use std::path::{Path, PathBuf};

#[derive(serde::Deserialize)]
struct Transcript {
    id: String,
    segments: Vec<Segment>,
}

#[derive(serde::Deserialize)]
struct Segment {
    speaker: String,
    text: String,
}

pub fn doppelgaenger() -> Result<()> {
    // step1_transcripts_to_text()?;
    // let text_file = step2_combine_texts()?;
    // let vocab = step3_build_vocab(text_file)?;
    let vocab = "./data/vocab-doppelgaenger.txt";
    step4_build_text_with_only_vocab(vocab)?;
    Ok(())
}

fn step1_transcripts_to_text() -> Result<()> {
    let mut files = Vec::new();
    for f in std::fs::read_dir("./data/doppelgaenger")? {
        let f = f?;
        if f.file_type()?.is_file() && f.path().extension() == Some("json".as_ref()) {
            files.push(f.path());
        }
    }

    for f in tqdm!(files.iter()) {
        let text = text_from_transcript_file(f)?;
        let output = f.with_extension(".txt");
        std::fs::write(output, text)?;
    }

    Ok(())
}

fn text_from_transcript_file(transcript_file: impl AsRef<Path>) -> Result<String> {
    let file = std::fs::File::open(transcript_file)?;
    let data: Transcript = serde_json::from_reader(file)?;

    let text = data
        .segments
        .into_iter()
        .map(|s| s.text.trim().to_string())
        .collect::<Vec<_>>()
        .join(" ");

    Ok(text)
}

fn step2_combine_texts() -> Result<PathBuf> {
    let mut files = Vec::new();
    for f in std::fs::read_dir("./data/doppelgaenger")? {
        let f = f?;
        if f.file_type()?.is_file() && f.path().extension() == Some("txt".as_ref()) {
            files.push(f.path());
        }
    }

    let corpus_file = "./data/doppelgaenger.txt";
    let mut corpus = std::fs::File::create(corpus_file)?;
    for f in tqdm!(files.iter()) {
        use std::io::Write;
        let text = std::fs::read_to_string(f)?;
        writeln!(corpus, "{}", text)?;
    }

    Ok(corpus_file.into())
}

lazy_static::lazy_static! {
    static ref WORDS_RE: regex::Regex = regex::Regex::new(r"([a-zA-ZäöüÄÖÜß]+)").unwrap();
}

fn step3_build_vocab(corpus_file: impl AsRef<Path>) -> Result<PathBuf> {
    // let corpus_file = std::path::PathBuf::from("./data/doppelgaenger.txt");
    let corpus = std::fs::read_to_string(corpus_file)?;

    let stop_words = std::fs::read_to_string("./data/stopwords-de.txt")?;

    let mut vocab = std::collections::HashSet::new();
    for word in tqdm!(WORDS_RE.find_iter(&corpus)) {
        let word = word.as_str().to_lowercase();
        if stop_words.contains(&word) {
            continue;
        }
        vocab.insert(word);
    }

    let vocab_path = "./data/vocab-doppelgaenger.txt";
    let mut vocab_file = std::fs::File::create(vocab_path)?;
    for word in vocab {
        use std::io::Write;
        writeln!(vocab_file, "{}", word)?;
    }

    Ok(vocab_path.into())
}

fn step4_build_text_with_only_vocab(vocab_file: impl AsRef<Path>) -> Result<()> {
    let vocab = std::fs::read_to_string(vocab_file)?
        .lines()
        .map(|s| s.to_string())
        .collect::<std::collections::HashSet<_>>();

    let mut files = Vec::new();
    for f in std::fs::read_dir("./data/doppelgaenger")? {
        let f = f?;
        if f.file_type()?.is_file() && f.path().extension() == Some("txt".as_ref()) {
            files.push(f.path());
        }
    }

    let only_vocab_texts_file = "./data/doppelgaenger-only-vocab.txt";
    let mut output_file = std::fs::File::create(only_vocab_texts_file)?;
    let mut last_word = String::new();
    for f in tqdm!(files.iter()) {
        use std::io::Write;
        let text = std::fs::read_to_string(f)?;

        for word in tqdm!(WORDS_RE.find_iter(&text)) {
            let word = word.as_str().to_lowercase();
            if vocab.contains(&word) && word != last_word {
                writeln!(output_file, "{}", word)?;
                last_word = word;
            }
        }
    }

    Ok(())
}

pub fn vocab_one_hot_encoded(vocab_file: impl AsRef<Path>) -> Result<(usize, Vec<u32>)> {
    info!("vocab_one_hot_encoded()");

    let text = std::fs::read_to_string(vocab_file)?;
    let words = text.split_whitespace().collect::<Vec<_>>();
    let n = words.len();
    let mut one_hot = vec![0; n];

    for (i, _word) in tqdm!(words.iter().enumerate()) {
        one_hot[i] = 1 << i;
    }

    Ok((n, one_hot))
}

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

const DATA_DIR: &str = "./data/doppelgaenger";

/// Contains the raw text from the transcripts
const CORPUS_FILE: &str = "./data/doppelgaenger.txt";

/// Contains the processed words, duplicates removed, stop words removed,
/// lowercased, etc. One word per line.
// const VOCAB_FILE: &str = "./data/vocab-doppelgaenger.txt";
const VOCAB_FILE: &str = "./data/tiny-vocab-doppelgaenger.txt";

const STOPWORDS_FILE: &str = "./data/stopwords-de.txt";

/// Contains the texts but only with words from the vocab. One word per line in
/// the order of the texts in the transcripts. Used for training.
const CONTEXT_FILE: &str = "./data/doppelgaenger-only-vocab.txt";

#[derive(Default)]
pub struct Vocab {
    pub words: Vec<String>,
    pub one_hot_encoded: Vec<u32>,
}

impl Vocab {
    pub fn build_from_scratch() -> Result<Self> {
        step1_transcripts_to_text()?;
        step2_combine_texts(CORPUS_FILE)?;
        step3_build_vocab(CORPUS_FILE, VOCAB_FILE)?;
        step4_build_text_with_only_vocab(VOCAB_FILE)?;

        Self::from_files()
    }

    pub fn from_files() -> Result<Self> {
        let (words, one_hot_encoded) = vocab_one_hot_encoded(VOCAB_FILE)?;
        Ok(Self {
            words,
            one_hot_encoded,
        })
    }

    pub fn n(&self) -> usize {
        self.words.len()
    }
}

fn step1_transcripts_to_text() -> Result<()> {
    let mut files = Vec::new();
    for f in std::fs::read_dir(DATA_DIR)? {
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

fn step2_combine_texts(corpus_file: impl AsRef<Path>) -> Result<()> {
    let mut files = Vec::new();
    for f in std::fs::read_dir(DATA_DIR)? {
        let f = f?;
        if f.file_type()?.is_file() && f.path().extension() == Some("txt".as_ref()) {
            files.push(f.path());
        }
    }

    let mut corpus = std::fs::File::create(corpus_file.as_ref())?;
    for f in tqdm!(files.iter()) {
        use std::io::Write;
        let text = std::fs::read_to_string(f)?;
        writeln!(corpus, "{}", text)?;
    }

    Ok(())
}

lazy_static::lazy_static! {
    static ref WORDS_RE: regex::Regex = regex::Regex::new(r"([a-zA-ZäöüÄÖÜß]+)").unwrap();
}

fn step3_build_vocab(corpus_file: impl AsRef<Path>, vocab_file: impl AsRef<Path>) -> Result<()> {
    // let corpus_file = std::path::PathBuf::from("./data/doppelgaenger.txt");
    let corpus = std::fs::read_to_string(corpus_file)?;

    let stop_words = std::fs::read_to_string(STOPWORDS_FILE)?;

    let mut vocab = std::collections::HashSet::new();
    for word in tqdm!(WORDS_RE.find_iter(&corpus)) {
        let word = word.as_str().to_lowercase();
        if stop_words.contains(&word) {
            continue;
        }
        vocab.insert(word);
    }

    let mut vocab_file = std::fs::File::create(vocab_file)?;
    for word in vocab {
        use std::io::Write;
        writeln!(vocab_file, "{}", word)?;
    }

    Ok(())
}

fn step4_build_text_with_only_vocab(vocab_file: impl AsRef<Path>) -> Result<()> {
    let vocab = std::fs::read_to_string(vocab_file)?
        .lines()
        .map(|s| s.to_string())
        .collect::<std::collections::HashSet<_>>();

    let mut files = Vec::new();
    for f in std::fs::read_dir(DATA_DIR)? {
        let f = f?;
        if f.file_type()?.is_file() && f.path().extension() == Some("txt".as_ref()) {
            files.push(f.path());
        }
    }

    let mut output_file = std::fs::File::create(CONTEXT_FILE)?;
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

fn vocab_one_hot_encoded(vocab_file: impl AsRef<Path>) -> Result<(Vec<String>, Vec<u32>)> {
    info!("vocab_one_hot_encoded()");

    let text = std::fs::read_to_string(vocab_file)?;
    let words = text
        .split_whitespace()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let n = words.len();
    let mut one_hot = vec![0; n];

    for i in tqdm!(0..n) {
        one_hot[i] = 1 << i;
    }

    Ok((words, one_hot))
}

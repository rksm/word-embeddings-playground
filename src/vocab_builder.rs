use anyhow::Result;
use kdam::tqdm;
use std::{collections::HashMap, path::Path};

#[allow(dead_code)]
#[derive(serde::Deserialize)]
struct Transcript {
    id: String,
    segments: Vec<Segment>,
}

#[allow(dead_code)]
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
// const CONTEXT_FILE: &str = "./data/context.txt";
const CONTEXT_FILE: &str = "./data/context-tiny.txt";

#[derive(Default)]
pub struct Vocab {
    pub words: Vec<String>,
    pub words_by_index: HashMap<String, usize>,
    pub context_word_indices: Vec<usize>,
}

impl Vocab {
    #[allow(dead_code)]
    pub fn build_from_scratch() -> Result<Self> {
        step1_transcripts_to_text()?;
        step2_combine_texts(CORPUS_FILE)?;
        step3_build_vocab(CORPUS_FILE, VOCAB_FILE)?;
        step4_build_text_with_only_vocab(VOCAB_FILE)?;

        Self::from_files()
    }

    #[allow(dead_code)]
    pub fn from_files() -> Result<Self> {
        let words = std::fs::read_to_string(VOCAB_FILE)?
            .lines()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let words_by_index = words
            .iter()
            .enumerate()
            .map(|(i, s)| (s.to_string(), i))
            .collect::<HashMap<_, _>>();

        // read the context file and map each word to its index in the vocab

        // dbg!(&words_by_index);
        let context = std::fs::read_to_string(CONTEXT_FILE)?;
        let context_words = context.lines().collect::<Vec<_>>();
        let context_word_indices = context_words
            .iter()
            .filter_map(|s| words_by_index.get(*s))
            .copied()
            .collect::<Vec<_>>();

        Ok(Self {
            words,
            words_by_index,
            context_word_indices,
        })
    }

    pub fn n(&self) -> usize {
        self.words.len()
    }

    pub fn context_n(&self) -> usize {
        self.context_word_indices.len()
    }

    /// Looks up the word at `index` in the context text. Returns the indices of the
    /// 2 words before and after. Also returns the index of the word itself.
    pub fn context(&self, index: usize) -> ([usize; 4], usize) {
        let n = self.context_n();
        let mut context = [0; 4];
        let lookup = |index: usize| -> usize { self.context_word_indices[index] };

        match index {
            0 => {
                context[2] = lookup(index + 1);
                context[3] = lookup(index + 2);
            }
            1 => {
                context[1] = lookup(index - 1);
                context[2] = lookup(index + 1);
                context[3] = lookup(index + 2);
            }
            _ if index == n - 1 => {
                context[0] = lookup(index - 2);
                context[1] = lookup(index - 1);
            }
            _ if index == n - 2 => {
                context[0] = lookup(index - 2);
                context[1] = lookup(index - 1);
                context[2] = lookup(index + 1);
            }
            _ => {
                context[0] = lookup(index - 2);
                context[1] = lookup(index - 1);
                context[2] = lookup(index + 1);
                context[3] = lookup(index + 2);
            }
        }

        (context, lookup(index))
    }

    pub fn word_lookup(&self, idx: usize) -> &str {
        &self.words[idx]
    }
}

fn step1_transcripts_to_text() -> Result<()> {
    let mut files = Vec::new();
    for f in std::fs::read_dir(DATA_DIR)?.take(1) {
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

use anyhow::Result;
use kdam::tqdm;
use rust_stemmers::{Algorithm, Stemmer};
use std::{collections::HashMap, path::Path};

use crate::dataset::CONTEXT_SIZE;

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

const DATA_DIR: &str = "./data/doppelgaenger/raw/";

/// Contains the raw text from the transcripts
const CORPUS_FILE: &str = "./data/doppelgaenger/doppelgaenger.txt";

/// Contains the processed words, duplicates removed, stop words removed,
/// lowercased, etc. One word per line.
const VOCAB_FILE: &str = "./data/doppelgaenger/vocab-doppelgaenger.txt";
// const VOCAB_FILE: &str = "./data/doppelgaenger/tiny-vocab-doppelgaenger.txt";

const STOPWORDS_FILE: &str = "./data/stopwords-de.txt";

/// Contains the texts but only with words from the vocab. One word per line in
/// the order of the texts in the transcripts. Used for training.
const CONTEXT_FILE: &str = "./data/doppelgaenger/context.txt";
// const CONTEXT_FILE: &str = "./data/doppelgaenger/context-tiny.txt";

pub struct Vocab {
    pub words: Vec<String>,
    pub words_by_index: HashMap<String, usize>,
    pub context_word_indices: Vec<usize>,
    pub stemmer: Stemmer,
    pub use_stemmer: bool,
}

impl Default for Vocab {
    fn default() -> Self {
        Self {
            words: Default::default(),
            words_by_index: Default::default(),
            context_word_indices: Default::default(),
            stemmer: Stemmer::create(Algorithm::German),
            use_stemmer: false,
        }
    }
}

impl Vocab {
    #[allow(dead_code)]
    pub fn build_from_scratch(use_stemmer: bool, use_stopwords: bool) -> Result<Self> {
        step1_transcripts_to_text()?;
        step2_combine_texts(CORPUS_FILE)?;
        step3_build_vocab(use_stemmer, use_stopwords, CORPUS_FILE, VOCAB_FILE)?;
        step4_build_text_with_only_vocab(use_stemmer, VOCAB_FILE)?;

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
            ..Default::default()
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
    pub fn context(&self, index: usize) -> ([u32; CONTEXT_SIZE], u32) {
        let mut context = [0; CONTEXT_SIZE];
        fill_context(&mut context, &self.context_word_indices, index);
        (context, self.context_word_indices[index] as u32)
    }

    pub fn word_lookup(&self, idx: usize) -> &str {
        &self.words[idx]
    }

    pub fn encode(&self, word: &str) -> Option<u32> {
        self.words_by_index
            .get(word)
            .or_else(|| {
                if self.use_stemmer {
                    let word = self.stemmer.stem(word);
                    self.words_by_index.get(word.as_ref())
                } else {
                    None
                }
            })
            .copied()
            .map(|i| i as u32)
    }

    pub fn context_to_word_index(&self, idx: u32) -> u32 {
        self.context_word_indices[idx as usize] as _
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
        let output = f.with_extension("txt");
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
        .join("\n");

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

fn step3_build_vocab(
    use_stemmer: bool,
    use_stopwords: bool,
    corpus_file: impl AsRef<Path>,
    vocab_file: impl AsRef<Path>,
) -> Result<()> {
    // let corpus_file = std::path::PathBuf::from("./data/doppelgaenger/doppelgaenger.txt");
    let corpus = std::fs::read_to_string(corpus_file)?;

    let stop_words = std::fs::read_to_string(STOPWORDS_FILE)?;

    let de_stemmer = Stemmer::create(Algorithm::German);
    let mut vocab = std::collections::HashMap::new();
    for word in tqdm!(WORDS_RE.find_iter(&corpus)) {
        let word = word.as_str().to_lowercase();
        if use_stopwords && stop_words.contains(&word) {
            continue;
        }
        if word.len() <= 2 {
            continue;
        }
        let word = if use_stemmer {
            de_stemmer.stem(word.as_str()).to_string()
        } else {
            word
        };
        *vocab.entry(word).or_insert(0) += 1;
    }

    const MIN_COUNT: usize = 3;
    let mut vocab_file = std::fs::File::create(vocab_file)?;
    for (word, count) in vocab {
        use std::io::Write;
        if count < MIN_COUNT {
            continue;
        }
        writeln!(vocab_file, "{}", word)?;
    }

    Ok(())
}

fn step4_build_text_with_only_vocab(use_stemmer: bool, vocab_file: impl AsRef<Path>) -> Result<()> {
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

    let de_stemmer = Stemmer::create(Algorithm::German);
    let mut output_file = std::fs::File::create(CONTEXT_FILE)?;
    let mut last_word = String::new();
    for f in tqdm!(files.iter()) {
        use std::io::Write;
        let text = std::fs::read_to_string(f)?;

        for word in tqdm!(WORDS_RE.find_iter(&text)) {
            let word = word.as_str().to_lowercase();
            let word = if use_stemmer {
                de_stemmer.stem(word.as_str()).to_string()
            } else {
                word
            };
            if vocab.contains(&word) && word != last_word {
                writeln!(output_file, "{}", word)?;
                last_word = word;
            }
        }
    }

    Ok(())
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// helper

fn fill_context(context: &mut [u32; CONTEXT_SIZE], vocab: &[usize], idx: usize) {
    let lower = &vocab[(idx.saturating_sub(CONTEXT_SIZE / 2))..idx];
    let upper = &vocab[idx + 1..(idx + 1 + CONTEXT_SIZE / 2).min(vocab.len())];

    let lower_offset = CONTEXT_SIZE / 2 - lower.len();
    for (i, w) in lower.iter().enumerate() {
        context[i + lower_offset] = *w as u32;
    }

    for (i, w) in upper.iter().enumerate() {
        context[i + CONTEXT_SIZE / 2] = *w as u32;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn fill_context() {
        let vocab = (20..30).collect::<Vec<_>>();

        let mut context = [0u32; crate::dataset::CONTEXT_SIZE];
        let idx = 8usize;
        super::fill_context(&mut context, &vocab, idx);
        assert_eq!(context, [24, 25, 26, 27, 29, 0, 0, 0]);

        let mut context = [0u32; crate::dataset::CONTEXT_SIZE];
        let idx = 2usize;
        super::fill_context(&mut context, &vocab, idx);
        assert_eq!(context, [0, 0, 20, 21, 23, 24, 25, 26]);

        let mut context = [0u32; crate::dataset::CONTEXT_SIZE];
        let idx = 5usize;
        super::fill_context(&mut context, &vocab, idx);
        assert_eq!(context, [21, 22, 23, 24, 26, 27, 28, 29]);
    }
}

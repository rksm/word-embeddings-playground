use rust_tokenizers::tokenizer::{BertTokenizer, MultiThreadedTokenizer, TruncationStrategy};

fn main() {
    a();
    b();
}

fn a() {
    // String Vector of Texts
    let texts: Vec<&str> =
        vec!["I couldn't believe that I could actually understand what I was reading."];

    // Tokens
    let vocab_file = concat!(env!("CARGO_MANIFEST_DIR"), "/data/doppelgaenger/vocab.txt");
    // let vocab_file = concat!(env!("CARGO_MANIFEST_DIR"), "/data/bert-base-uncased-vocab.txt");
    let tokenizer: BertTokenizer =
        BertTokenizer::from_file(vocab_file, true, true).expect("Error while loading vocab file");
    let tokenized_input = tokenizer.encode_list(&texts, 128, &TruncationStrategy::LongestFirst, 0);

    println!("{:?}", tokenized_input);
}

fn b() {
    use rust_tokenizers::adapters::Example;
    use rust_tokenizers::tokenizer::Tokenizer;
    use rust_tokenizers::vocab::{BertVocab, Vocab};

    let vocab_path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/doppelgaenger/vocab.txt");
    let vocab = BertVocab::from_file(vocab_path).expect("Unable to build vocab");
    let lowercase: bool = true;
    let strip_accents: bool = true;

    let test_sentence = Example::new_from_string("This is a sample sentence to be tokenized");
    let bert_tokenizer: BertTokenizer =
        BertTokenizer::from_existing_vocab(vocab, lowercase, strip_accents);

    println!(
        "{:?}",
        bert_tokenizer.encode(
            &test_sentence.sentence_1,
            None,
            128,
            &TruncationStrategy::LongestFirst,
            0
        )
    );
}

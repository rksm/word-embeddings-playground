use rust_stemmers::{Algorithm, Stemmer};

fn main() {
    let de_stemmer = Stemmer::create(Algorithm::German);
    dbg!(de_stemmer.stem("institutionelle"));
    dbg!(de_stemmer.stem("kaputter"));
    dbg!(de_stemmer.stem("heißt"));

    dbg!(de_stemmer.stem("schönes"));
    dbg!(de_stemmer.stem("wetter"));
    dbg!(de_stemmer.stem("abverkaufsstark"));
    dbg!(de_stemmer.stem("ausgehst"));
    dbg!(de_stemmer.stem("kampagnen"));
    dbg!(de_stemmer.stem("wertvoll"));
    dbg!(de_stemmer.stem("überlaufen"));
    dbg!(de_stemmer.stem("angeschlossenen"));
    dbg!(de_stemmer.stem("angeschlossen"));
}

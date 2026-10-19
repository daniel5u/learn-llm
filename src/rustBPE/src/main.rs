mod byte_map;
mod tokenizer;

use crate::tokenizer::Tokenizer;

fn main() {
    // [ Test of string-byte relations. ]
    // let test = String::from("你好, Hello! lo");
    //
    // let test_bytes = test.as_bytes();
    //
    // println!("你好 in bytes is: {:?}", test_bytes);
    //
    // println!("And that will be transferred into:");
    //
    // println!("{}", visualize(test_bytes));
    //
    // [ Test of training a tokenizer and save it.]
    let text = std::fs::read_to_string("src/tinyShakeSpeare.txt").expect("Cannot read file");

    let mut test_tokenizer = Tokenizer::new();

    test_tokenizer.train(&text, 4000);
    //
    // test_tokenizer.show_vocabulary();

    test_tokenizer.save();
    //
    // [ Test of loading a tokenizer from files.]
    // let test_tokenizer = Tokenizer::load("vocabulary.ron", "merges.ron");
    //
    // test_tokenizer.show_vocabulary();
    //
    // [ Test of encoding a string into TokenId.]
    // let test_tokenizer = Tokenizer::load("vocabulary.ron", "merges.ron");

    let test_string = String::from(
        "I say unto you, what he hath done famously, he did
      it to that end: though soft-conscienced men can be
      content to say it was for his country he did it to
      please his mother and to be partly proud; which he
      is, even till the altitude of his virtue.",
    );

    let encoded = test_tokenizer.encode(&test_string);

    test_tokenizer.visualize_encoded_ids(&encoded);

    // let decoded = test_tokenizer.decode(&encoded);

    // println!("decoded from the encoded: {decoded}");
}

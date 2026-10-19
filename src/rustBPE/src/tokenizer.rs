use serde::{Deserialize, Serialize};
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
};

use crate::byte_map::visualize;
use indicatif::{ProgressBar, ProgressStyle};

type TokenId = u32;
type Pair = (TokenId, TokenId);

#[derive(Debug, Deserialize, Serialize)]
pub struct Tokenizer {
    merges: HashMap<Pair, TokenId>,
    vocab: Vec<Vec<u8>>,
}

const CONTRACTION: [&str; 7] = ["ll", "ve", "re", "s", "d", "m", "t"];

fn is_newline(c: char) -> bool {
    c == '\r' || c == '\n'
}

fn char_at(text: &str, i: usize) -> Option<char> {
    text.get(i..)?.chars().next()
}

fn skip_while(text: &str, start: usize, pred: impl Fn(char) -> bool) -> usize {
    text[start..]
        .char_indices()
        .find(|&(_, c)| !pred(c))
        .map_or(text.len(), |(k, _)| start + k)
}

fn is_other(c: char) -> bool {
    !(c.is_whitespace() || c.is_alphabetic() || c.is_numeric())
}
impl Tokenizer {
    pub fn new() -> Self {
        let mut default_vocab = Vec::new();

        for i in 0..=255 {
            default_vocab.push(vec![i as u8]);
        }

        Self {
            merges: HashMap::new(),
            vocab: default_vocab,
        }
    }

    pub fn train(&mut self, train_text: &str, vocab_size: usize) {
        // Reset the tokeknizer.
        *self = Self::new();

        let mut word_count: HashMap<&str, u64> = HashMap::new();
        let text_words = Self::pre_tokenize(train_text);
        for word in text_words {
            *word_count.entry(word).or_insert(0) += 1;
        }

        let mut ids: Vec<TokenId> = Vec::new();
        let mut weight: Vec<i64> = Vec::new();
        let mut prev: Vec<Option<u32>> = Vec::new();
        let mut next: Vec<Option<u32>> = Vec::new();

        for (word, word_cnt) in word_count {
            let start = ids.len();
            let bytes = word.as_bytes();
            for (k, &b) in bytes.iter().enumerate() {
                let i = (start + k) as u32;
                ids.push(TokenId::from(b));
                weight.push(word_cnt as i64);
                prev.push((k > 0).then(|| i - 1));
                next.push((k + 1 < bytes.len()).then(|| i + 1));
            }
        }

        let n = ids.len();
        let mut alive: Vec<bool> = vec![true; n];

        let mut occ: HashMap<Pair, Vec<u32>> = HashMap::new();
        let mut counts: HashMap<Pair, i64> = HashMap::new();

        for i in 0..n {
            if let Some(j) = next[i] {
                let pair = (ids[i], ids[j as usize]);

                occ.entry(pair).or_default().push(i as u32);
                *counts.entry(pair).or_insert(0) += weight[i];
            }
        }

        let mut count_heap: BinaryHeap<(i64, Reverse<Pair>)> =
            counts.iter().map(|(&p, &c)| (c, Reverse(p))).collect();

        let mut touched: Vec<Pair> = Vec::new();

        let pb = ProgressBar::new(vocab_size as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] {bar:30.cyan/blue} {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏  "),
        );

        while self.vocab.len() < vocab_size {
            let pair_to_merge = loop {
                let Some((count, Reverse(pair))) = count_heap.pop() else {
                    pb.abandon_with_message("No pair left");
                    return;
                };

                if count == counts.get(&pair).copied().unwrap_or(0) && count > 0 {
                    break pair;
                }
            };

            let (left, right) = pair_to_merge;
            let new_id = self.add_new_vocab(pair_to_merge);

            let appears = occ.remove(&pair_to_merge).unwrap_or_default();
            counts.remove(&pair_to_merge);

            touched.clear();

            for appear in appears {
                let index = appear as usize;

                if !alive[index] || ids[index] != left {
                    continue;
                }

                let Some(j) = next[index] else { continue };
                let j = j as usize;
                if !alive[j] || ids[j] != right {
                    continue;
                };

                let p = prev[index];
                let n = next[j];

                // Update: count, count_heap, alive
                if let Some(p) = p {
                    let left_pair = (ids[p as usize], left);
                    if let Some(c) = counts.get_mut(&left_pair) {
                        *c -= weight[p as usize];
                        touched.push(left_pair);
                    }
                }

                if let Some(n) = n {
                    let right_pair = (right, ids[n as usize]);
                    if let Some(c) = counts.get_mut(&right_pair) {
                        *c -= weight[n as usize];
                        touched.push(right_pair);
                    }
                }

                ids[index] = new_id;
                alive[j] = false;
                next[index] = n;

                if let Some(n) = n {
                    prev[n as usize] = Some(appear);
                }

                if let Some(p) = p {
                    let left_pair = (ids[p as usize], new_id);
                    *counts.entry(left_pair).or_insert(0) += weight[p as usize];
                    occ.entry(left_pair).or_default().push(p);
                    touched.push(left_pair);
                }

                if let Some(n) = n {
                    let right_pair = (new_id, ids[n as usize]);
                    *counts.entry(right_pair).or_insert(0) += weight[n as usize];
                    occ.entry(right_pair).or_default().push(appear);
                    touched.push(right_pair);
                }
            }

            touched.sort_unstable();
            touched.dedup();

            for &pair in &touched {
                if let Some(&c) = counts.get(&pair)
                    && c > 0
                {
                    count_heap.push((c, Reverse(pair)));
                }
            }

            pb.inc(1);
        }
        pb.finish_with_message("Training done");
    }

    fn add_new_vocab(&mut self, pair: Pair) -> TokenId {
        let (left, right) = pair;

        let new_id = TokenId::try_from(self.vocab.len()).expect("Too many tokens!");

        let mut new_bytes = self.vocab[left as usize].clone();
        new_bytes.extend_from_slice(&self.vocab[right as usize]);

        self.vocab.push(new_bytes);
        self.merges.insert((left, right), new_id);

        new_id
    }

    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        let text_words = Self::pre_tokenize(text);
        let mut ids: Vec<TokenId> = Vec::new();
        let mut prev: Vec<Option<u32>> = Vec::new();
        let mut next: Vec<Option<u32>> = Vec::new();

        for word in text_words {
            let start = ids.len();
            let bytes = word.as_bytes();
            for (k, &b) in bytes.iter().enumerate() {
                let i = (start + k) as u32;
                ids.push(TokenId::from(b));
                prev.push((k > 0).then(|| i - 1));
                next.push((k + 1 < bytes.len()).then(|| i + 1));
            }
        }

        let n = ids.len();
        let mut alive: Vec<bool> = vec![true; n];

        let pb = ProgressBar::new(n as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] {bar:30.cyan/blue} {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏  "),
        );

        let mut pair_heap: BinaryHeap<Reverse<(TokenId, u32)>> = (0..n - 1)
            .filter_map(|i| {
                let j = next[i]? as usize;
                self.merges
                    .get(&(ids[i], ids[j]))
                    .map(|&id| Reverse((id, i as u32)))
            })
            .collect();

        while let Some(Reverse((new_id, slot))) = pair_heap.pop() {
            let index = slot as usize;

            if !alive[index] {
                continue;
            }

            let Some(jndex) = next[index] else { continue };
            let jndex = jndex as usize;
            if !alive[jndex] {
                continue;
            };
            match self.merges.get(&(ids[index], ids[jndex])) {
                Some(&id) if id == new_id => {}
                _ => continue,
            }

            let p = prev[index];
            let n = next[jndex];

            ids[index] = new_id;
            alive[jndex] = false;
            next[index] = n;
            if let Some(n) = n {
                prev[n as usize] = Some(slot);
            }

            if let Some(p) = p
                && let Some(&id) = self.merges.get(&(ids[p as usize], new_id))
            {
                pair_heap.push(Reverse((id, p)));
            }

            if let Some(n) = n
                && let Some(&id) = self.merges.get(&(new_id, ids[n as usize]))
            {
                pair_heap.push(Reverse((id, slot)));
            }

            pb.inc(1);
        }

        (0..n).filter(|&i| alive[i]).map(|i| ids[i]).collect()
    }

    pub fn decode(&self, ids: &[TokenId]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&id| self.vocab[id as usize].iter())
            .copied()
            .collect();

        String::from_utf8(bytes).expect("Cannot decode the ids!")
    }

    pub fn show_vocabulary(&self) {
        for (index, word) in self.vocab.iter().enumerate() {
            print!("{}: {}\t", index, visualize(word));
        }
        println!();
    }

    pub fn save(&self) {
        let vocab_ron = ron::ser::to_string_pretty(&self.vocab, ron::ser::PrettyConfig::default())
            .expect("Cannot serialize the vocabulary!");
        let merges_ron =
            ron::ser::to_string_pretty(&self.merges, ron::ser::PrettyConfig::default())
                .expect("Cannot serialize the merges!");

        std::fs::write("vocabulary.ron", vocab_ron).expect("Cannot write vocabulary!");
        std::fs::write("merges.ron", merges_ron).expect("Cannot write merges!");
    }

    pub fn load(vocabulary_file_path: &str, merges_file_path: &str) -> Self {
        let vocab_str =
            std::fs::read_to_string(vocabulary_file_path).expect("Cannot read Vocabulary file");
        let merges_str =
            std::fs::read_to_string(merges_file_path).expect("Cannot read merges file");

        let vocab: Vec<Vec<u8>> = ron::from_str(&vocab_str)
            .expect("Cannot deserialize the vocabulary file. Check the format.");
        let merges: HashMap<Pair, TokenId> = ron::from_str(&merges_str)
            .expect("Cannot deserialize the vocabulary file. Check the format.");

        Self { vocab, merges }
    }

    pub fn visualize_encoded_ids(&self, ids: &[TokenId]) {
        ids.iter()
            .for_each(|id| print!("[{:?}]", visualize(&self.vocab[*id as usize])));
        println!();
    }

    fn pre_tokenize(text: &str) -> Vec<&str> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < text.len() {
            let j = Self::match_at(text, i);
            out.push(&text[i..j]);
            i = j;
        }
        out
    }

    fn match_at(text: &str, i: usize) -> usize {
        let bytes = text.as_bytes();
        let c = char_at(text, i).expect("i must be < text.len()");
        let after_c = i + c.len_utf8();

        // Rule 1: 'll, 've, 're, 's, 'd, 'm, 't
        if c == '\'' {
            for con in CONTRACTION {
                if let Some(b) = bytes.get(after_c..after_c + con.len())
                    && b.eq_ignore_ascii_case(con.as_bytes())
                {
                    return after_c + con.len();
                }
            }
        }

        // Rule 2: find a complete word
        let mut j = i;
        if !is_newline(c)
            && !c.is_alphabetic()
            && !c.is_numeric()
            && char_at(text, after_c).is_some_and(char::is_alphabetic)
        {
            j = after_c;
        }
        if char_at(text, j).is_some_and(char::is_alphabetic) {
            return skip_while(text, j, char::is_alphabetic);
        }

        // Rule 3: up to 3 digits
        if c.is_numeric() {
            let mut j = i;
            for _ in 0..3 {
                match char_at(text, j) {
                    Some(d) if d.is_numeric() => j += d.len_utf8(),
                    _ => break,
                }
            }
            return j;
        }

        // Rule 4: optinal space + symbols + trailing newlines
        let j = if c == ' ' { after_c } else { i };
        if char_at(text, j).is_some_and(is_other) {
            let j = skip_while(text, j, is_other);
            return skip_while(text, j, is_newline);
        }

        // Rule 5-7: whitespace
        let end = skip_while(text, i, char::is_whitespace);
        let run = &text[i..end];
        if let Some(k) = run.rfind(['\r', '\n']) {
            return i + k + 1;
        }
        if end == text.len() || run.chars().count() == 1 {
            return end;
        }
        end - run.chars().next_back().unwrap().len_utf8()
    }
}

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::case_space::{
    case_space_encode, CaseSpaceDecoder, CH_FIRST_UPPER, CH_NO_SPACE, CH_TO_UPPER,
};
use crate::dictionary::Dictionary;
use crate::error::PreprocessError;

const NS: usize = 256;
const MAX_WORDS_PER_ITER: usize = 100;
const SUBST_COST: f64 = 7.0;
const TOT_FREQ_RED_BITS: f64 = 1.3;

/// Ordering applied to the final dictionary entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortOrder {
    /// Keep base bytes `0..256` followed by learned merges in discovery order.
    #[default]
    Discovery,
    /// Sort all symbols by decreasing corpus frequency (`-s` flag in `preprocess.c`).
    ByFrequency,
}

/// Parameters controlling dictionary induction during preprocessing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeConfig {
    /// Approximate maximum vocabulary size, including the 256 base byte symbols.
    pub max_words: usize,
    /// Minimum frequency required for a merged word to be added and retained.
    pub min_freq: u32,
    /// Output vocabulary ordering.
    pub sort_order: SortOrder,
    /// Optional path where frequency/coverage debug info is written (`-D` in `preprocess.c`).
    pub debug_dict_path: Option<PathBuf>,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        Self {
            max_words: 4097,
            min_freq: 5,
            sort_order: SortOrder::Discovery,
            debug_dict_path: None,
        }
    }
}

/// Summary statistics from a preprocessing run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncodeReport {
    /// Input byte count before preprocessing.
    pub input_bytes: usize,
    /// Symbol count immediately after case/space encoding.
    pub case_space_symbols: usize,
    /// Final vocabulary size (`256 + learned_words`).
    pub vocab_size: usize,
    /// Final emitted token count.
    pub token_count: usize,
    /// Zero-order symbol entropy in bits.
    pub entropy_bits: f64,
}

#[derive(Debug, Clone)]
struct WordEntry {
    pair: [u16; 2],
    freq: u32,
    score: f32,
}

#[derive(Debug, Default)]
struct WordList {
    words: Vec<WordEntry>,
    pair_to_idx: HashMap<[u16; 2], usize>,
}

impl WordList {
    fn find(&self, pair: [u16; 2]) -> Option<usize> {
        self.pair_to_idx.get(&pair).copied()
    }

    fn find_or_insert(&mut self, pair: [u16; 2]) -> usize {
        if let Some(&idx) = self.pair_to_idx.get(&pair) {
            return idx;
        }
        let idx = self.words.len();
        self.words.push(WordEntry {
            pair,
            freq: 0,
            score: 0.0,
        });
        self.pair_to_idx.insert(pair, idx);
        idx
    }
}

fn expand_symbol(words: &[WordEntry], code: usize, out: &mut Vec<u8>) {
    if code < NS {
        out.push(code as u8);
    } else {
        let entry = &words[code - NS];
        expand_symbol(words, usize::from(entry.pair[0]), out);
        expand_symbol(words, usize::from(entry.pair[1]), out);
    }
}

fn get_n_bits(code: usize, cw: &WordList, char_freq: &[u32; NS], tot_freq: usize) -> f32 {
    if code < NS {
        -(((f64::from(char_freq[code]) / tot_freq as f64).log2()) as f32)
    } else {
        let p = &cw.words[code - NS];
        if p.freq == 0 {
            get_n_bits(usize::from(p.pair[0]), cw, char_freq, tot_freq)
                + get_n_bits(usize::from(p.pair[1]), cw, char_freq, tot_freq)
        } else {
            -(((f64::from(p.freq) / tot_freq as f64).log2()) as f32)
        }
    }
}

fn compute_score(entry: &WordEntry, cw: &WordList, char_freq: &[u32; NS], tot_freq: usize) -> f32 {
    if entry.freq <= 1 {
        return -1.0;
    }
    let old_bits = (get_n_bits(usize::from(entry.pair[0]), cw, char_freq, tot_freq)
        + get_n_bits(usize::from(entry.pair[1]), cw, char_freq, tot_freq))
        * (entry.freq as f32);
    let new_bits = ((-(f64::from(entry.freq) / tot_freq as f64).log2() + TOT_FREQ_RED_BITS)
        * f64::from(entry.freq)
        + SUBST_COST) as f32;
    old_bits - new_bits
}

fn select_best_words(
    ws: &mut WordList,
    max_to_add: usize,
    cw: &mut WordList,
    char_freq: &[u32; NS],
    tot_freq: usize,
    min_freq: u32,
) -> usize {
    for entry in &mut ws.words {
        entry.score = compute_score(entry, cw, char_freq, tot_freq);
    }

    ws.words.sort_by(|a, b| {
        if a.score > b.score {
            Ordering::Less
        } else if a.score < b.score {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });

    let bitmap_len = NS + cw.words.len();
    let mut bitmap_start = vec![false; bitmap_len];
    let mut bitmap_end = vec![false; bitmap_len];

    let mut added = 0usize;
    for entry in &ws.words {
        if entry.score <= 0.0 || entry.freq < min_freq {
            break;
        }
        let a = usize::from(entry.pair[0]);
        let b = usize::from(entry.pair[1]);
        if bitmap_end[a] || bitmap_start[b] {
            continue;
        }
        bitmap_start[a] = true;
        bitmap_end[b] = true;
        cw.find_or_insert(entry.pair);
        added += 1;
        if added >= max_to_add {
            break;
        }
    }
    added
}

fn compute_word_freq(s: &mut WordList, char_freq: &mut [u32; NS], buf: &[u16]) {
    for entry in &mut s.words {
        entry.freq = 0;
    }
    char_freq.fill(0);
    for &c in buf {
        let c_usize = usize::from(c);
        if c_usize >= NS {
            s.words[c_usize - NS].freq += 1;
        } else {
            char_freq[c_usize] += 1;
        }
    }
}

fn out_word(obuf: &mut Vec<u16>, s: &WordList, c: u16) {
    let c_usize = usize::from(c);
    if c_usize < NS {
        obuf.push(c);
    } else {
        let p = &s.words[c_usize - NS];
        if p.freq == 0 {
            out_word(obuf, s, p.pair[0]);
            out_word(obuf, s, p.pair[1]);
        } else {
            obuf.push(c);
        }
    }
}

fn update_word_freq(
    s: &mut WordList,
    char_freq: &mut [u32; NS],
    buf: &mut Vec<u16>,
    min_freq: u32,
) -> usize {
    compute_word_freq(s, char_freq, buf);

    let mut word_count = 0usize;
    for entry in &mut s.words {
        if entry.freq >= min_freq {
            word_count += 1;
        } else {
            entry.freq = 0;
        }
    }
    if word_count == s.words.len() {
        return word_count;
    }

    let mut obuf = Vec::with_capacity(buf.len());
    for &sym in buf.iter() {
        out_word(&mut obuf, s, sym);
    }
    *buf = obuf;
    compute_word_freq(s, char_freq, buf);
    word_count
}

fn compute_entropy(s: &WordList, char_freq: &[u32; NS], buf_len: usize) -> f64 {
    if buf_len == 0 {
        return 0.0;
    }
    let total = buf_len as f64;
    let mut n_bits = 0.0f64;
    for &freq in char_freq {
        if freq != 0 {
            let f = f64::from(freq);
            n_bits += -(f / total).log2() * f;
        }
    }
    for entry in &s.words {
        if entry.freq > 0 {
            let f = f64::from(entry.freq);
            n_bits += -(f / total).log2() * f;
        }
    }
    n_bits
}

fn sort_words(s: &WordList, char_freq: &[u32; NS], order: SortOrder) -> Vec<usize> {
    let total_slots = NS + s.words.len();
    let mut tab = Vec::with_capacity(total_slots);
    for i in 0..total_slots {
        if i >= NS && s.words[i - NS].freq == 0 {
            continue;
        }
        tab.push(i);
    }
    if order == SortOrder::ByFrequency {
        tab.sort_by(|&c1, &c2| {
            let freq1 = if c1 < NS {
                char_freq[c1]
            } else {
                s.words[c1 - NS].freq
            };
            let freq2 = if c2 < NS {
                char_freq[c2]
            } else {
                s.words[c2 - NS].freq
            };
            freq2.cmp(&freq1)
        });
    }
    tab
}

fn write_debug_word(out: &mut impl Write, words: &[WordEntry], code: usize) -> std::io::Result<()> {
    if code < NS {
        let byte = code as u8;
        match byte {
            b'\n' => out.write_all(b"\\n"),
            b'\\' => out.write_all(b"\\\\"),
            CH_TO_UPPER => out.write_all(b"\\u"),
            CH_FIRST_UPPER => out.write_all(b"\\c"),
            CH_NO_SPACE => out.write_all(b"\\s"),
            other => out.write_all(&[other]),
        }
    } else {
        let entry = &words[code - NS];
        write_debug_word(out, words, usize::from(entry.pair[0]))?;
        write_debug_word(out, words, usize::from(entry.pair[1]))
    }
}

fn save_words_debug(
    path: &Path,
    s: &WordList,
    char_freq: &[u32; NS],
    tot_freq: usize,
    tab: &[usize],
) -> Result<(), PreprocessError> {
    let file = File::create(path)?;
    let mut out = BufWriter::new(file);
    writeln!(out, "{:>7} {:>5} WORD", "FREQ", "CUM%")?;
    let mut sum = 0u64;
    let denom = tot_freq.max(1) as f64;
    for &c in tab {
        let freq = if c < NS {
            char_freq[c]
        } else {
            s.words[c - NS].freq
        };
        sum += u64::from(freq);
        write!(out, "{freq:>7} {:>5.1} '", (sum as f64) / denom * 100.0)?;
        write_debug_word(&mut out, &s.words, c)?;
        writeln!(out, "'")?;
    }
    out.flush()?;
    Ok(())
}

fn emit_token_ids(
    code: usize,
    words: &[WordEntry],
    convert_table: &[u16],
    out_tokens: &mut Vec<u16>,
) {
    if code < NS {
        out_tokens.push(convert_table[code]);
    } else {
        let idx = code - NS;
        let entry = &words[idx];
        if entry.freq == 0 {
            emit_token_ids(usize::from(entry.pair[0]), words, convert_table, out_tokens);
            emit_token_ids(usize::from(entry.pair[1]), words, convert_table, out_tokens);
        } else {
            out_tokens.push(convert_table[code]);
        }
    }
}

fn fnv1a_u16(slice: &[u16]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &v in slice {
        hash ^= u64::from(v);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// Runs Fabrice Bellard's bigram dictionary induction and encodes `input` into token IDs.
///
/// # Errors
///
/// Returns [`PreprocessError::InvalidMaxWords`] if `config.max_words < 256` or `> 65536`,
/// or [`PreprocessError::Io`] if `config.debug_dict_path` cannot be written.
pub fn encode_bytes(
    input: &[u8],
    config: &EncodeConfig,
) -> Result<(Dictionary, Vec<u16>, EncodeReport), PreprocessError> {
    if config.max_words < NS || config.max_words > usize::from(u16::MAX) + 1 {
        return Err(PreprocessError::InvalidMaxWords(config.max_words));
    }

    let mut buf = case_space_encode(input);
    let case_space_symbols = buf.len();
    let mut s = WordList::default();
    let mut char_freq = [0u32; NS];
    compute_word_freq(&mut s, &mut char_freq, &buf);

    let max_learned = config.max_words - NS;
    let mut word_count = 0usize;
    let mut entropy_bits = compute_entropy(&s, &char_freq, buf.len());
    let mut seen_states = HashSet::new();
    seen_states.insert((s.words.len(), word_count, buf.len(), fnv1a_u16(&buf)));

    while word_count < max_learned {
        if buf.len() < 2 {
            break;
        }
        let mut ws = WordList::default();
        for pair in buf.windows(2) {
            let idx = ws.find_or_insert([pair[0], pair[1]]);
            ws.words[idx].freq += 1;
        }
        let added = select_best_words(
            &mut ws,
            MAX_WORDS_PER_ITER.min(max_learned - word_count),
            &mut s,
            &char_freq,
            buf.len(),
            config.min_freq,
        );
        if added == 0 {
            break;
        }

        let mut next_buf = Vec::with_capacity(buf.len());
        let mut i = 0usize;
        while i < buf.len() {
            if i + 1 < buf.len() {
                if let Some(word_idx) = s.find([buf[i], buf[i + 1]]) {
                    next_buf.push((NS + word_idx) as u16);
                    i += 2;
                    continue;
                }
            }
            next_buf.push(buf[i]);
            i += 1;
        }
        buf = next_buf;

        let word_count_prev = word_count;
        word_count = update_word_freq(&mut s, &mut char_freq, &mut buf, config.min_freq);
        entropy_bits = compute_entropy(&s, &char_freq, buf.len());
        let state = (s.words.len(), word_count, buf.len(), fnv1a_u16(&buf));
        if word_count >= max_learned || word_count == word_count_prev || !seen_states.insert(state)
        {
            break;
        }
    }

    let tab_sort = sort_words(&s, &char_freq, config.sort_order);
    if let Some(debug_path) = &config.debug_dict_path {
        save_words_debug(debug_path, &s, &char_freq, buf.len(), &tab_sort)?;
    }

    let mut dict_words = Vec::with_capacity(tab_sort.len());
    let mut convert_table = vec![u16::MAX; NS + s.words.len()];
    for (new_id, &orig_code) in tab_sort.iter().enumerate() {
        convert_table[orig_code] = new_id as u16;
        let mut word_bytes = Vec::new();
        expand_symbol(&s.words, orig_code, &mut word_bytes);
        dict_words.push(word_bytes);
    }

    let mut out_tokens = Vec::with_capacity(buf.len());
    for &code in &buf {
        emit_token_ids(usize::from(code), &s.words, &convert_table, &mut out_tokens);
    }

    let dict = Dictionary::from_words(dict_words);
    let report = EncodeReport {
        input_bytes: input.len(),
        case_space_symbols,
        vocab_size: dict.len(),
        token_count: out_tokens.len(),
        entropy_bits,
    };
    Ok((dict, out_tokens, report))
}

/// Preprocesses `in_path`, writing the learned dictionary to `dict_path` and big-endian tokens to `out_path`.
///
/// # Errors
///
/// Returns [`PreprocessError`] on invalid parameters or I/O failures.
pub fn encode_file(
    in_path: impl AsRef<Path>,
    out_path: impl AsRef<Path>,
    dict_path: impl AsRef<Path>,
    config: &EncodeConfig,
) -> Result<EncodeReport, PreprocessError> {
    let input = std::fs::read(in_path)?;
    let (dict, tokens, report) = encode_bytes(&input, config)?;
    dict.save(dict_path)?;
    write_tokens_be16(out_path, &tokens)?;
    Ok(report)
}

#[derive(Debug, Default)]
struct TrieNode {
    token: Option<u16>,
    children: Vec<(u8, usize)>,
}

/// Encodes `input` greedily (longest match first) using an existing [`Dictionary`].
///
/// Useful when compressing new text with a fixed vocabulary such as `assets/enwik9.voc`.
///
/// # Errors
///
/// Returns [`PreprocessError::MissingByteSymbol`] if `dict` does not contain all 256
/// single-byte base entries required as fallback.
pub fn encode_with_dictionary(
    input: &[u8],
    dict: &Dictionary,
) -> Result<Vec<u16>, PreprocessError> {
    let mut byte_tokens = [None::<u16>; 256];
    let mut nodes = vec![TrieNode::default()];

    for (idx, word) in dict.words().iter().enumerate() {
        let token = idx as u16;
        if word.len() == 1 && byte_tokens[usize::from(word[0])].is_none() {
            byte_tokens[usize::from(word[0])] = Some(token);
        }
        let mut node_idx = 0usize;
        for &b in word {
            let next_idx = nodes[node_idx]
                .children
                .iter()
                .find(|&&(ch, _)| ch == b)
                .map(|&(_, child)| child);
            node_idx = match next_idx {
                Some(child) => child,
                None => {
                    let child = nodes.len();
                    nodes.push(TrieNode::default());
                    nodes[node_idx].children.push((b, child));
                    child
                }
            };
        }
        if nodes[node_idx].token.is_none() {
            nodes[node_idx].token = Some(token);
        }
    }

    for (b, slot) in byte_tokens.iter().enumerate() {
        if slot.is_none() {
            return Err(PreprocessError::MissingByteSymbol(b as u8));
        }
    }

    let cs_u16 = case_space_encode(input);
    let cs: Vec<u8> = cs_u16.into_iter().map(|v| v as u8).collect();
    let mut tokens = Vec::with_capacity(cs.len() / 2 + 1);
    let mut pos = 0usize;

    while pos < cs.len() {
        let mut node_idx = 0usize;
        let mut best_token = byte_tokens[usize::from(cs[pos])]
            .ok_or_else(|| PreprocessError::MissingByteSymbol(cs[pos]))?;
        let mut best_len = 1usize;

        let mut cur = pos;
        while cur < cs.len() {
            let ch = cs[cur];
            let Some(&(_, next_idx)) = nodes[node_idx].children.iter().find(|&&(c, _)| c == ch)
            else {
                break;
            };
            node_idx = next_idx;
            cur += 1;
            if let Some(tok) = nodes[node_idx].token {
                best_token = tok;
                best_len = cur - pos;
            }
        }

        tokens.push(best_token);
        pos += best_len;
    }

    Ok(tokens)
}

/// Decodes a sequence of token IDs back into the exact original source bytes.
///
/// # Errors
///
/// Returns [`PreprocessError::InvalidSymbol`] if any token ID is `>= dict.len()`.
pub fn decode_tokens(tokens: &[u16], dict: &Dictionary) -> Result<Vec<u8>, PreprocessError> {
    let mut out = Vec::with_capacity(tokens.len() * 2);
    let mut decoder = CaseSpaceDecoder::new();
    for &tok in tokens {
        let Some(bytes) = dict.get(tok) else {
            return Err(PreprocessError::InvalidSymbol {
                symbol: tok,
                vocab_size: dict.len(),
            });
        };
        // Writing to Vec<u8> is infallible.
        let _ = decoder.push_bytes(bytes, &mut out);
    }
    Ok(out)
}

/// Streams a big-endian `u16` token file back to its original raw file using `dict_path`.
///
/// # Errors
///
/// Returns [`PreprocessError`] on I/O error, odd file length, or out-of-range symbol ID.
pub fn decode_file(
    in_path: impl AsRef<Path>,
    out_path: impl AsRef<Path>,
    dict_path: impl AsRef<Path>,
) -> Result<usize, PreprocessError> {
    let dict = Dictionary::load(dict_path)?;
    let in_file = File::open(in_path)?;
    let in_len = in_file.metadata()?.len();
    if in_len % 2 != 0 {
        return Err(PreprocessError::OddTokenStreamLength(in_len));
    }

    let mut reader = BufReader::with_capacity(256 * 1024, in_file);
    let out_file = File::create(out_path)?;
    let mut writer = BufWriter::with_capacity(256 * 1024, out_file);
    let mut decoder = CaseSpaceDecoder::new();
    let mut pair = [0u8; 2];
    let mut count = 0usize;

    loop {
        match reader.read_exact(&mut pair) {
            Ok(()) => {
                let tok = u16::from_be_bytes(pair);
                let Some(word) = dict.get(tok) else {
                    return Err(PreprocessError::InvalidSymbol {
                        symbol: tok,
                        vocab_size: dict.len(),
                    });
                };
                decoder.push_bytes(word, &mut writer)?;
                count += 1;
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
    }

    writer.flush()?;
    Ok(count)
}

/// Writes `tokens` as big-endian `u16` pairs to `path`.
///
/// # Errors
///
/// Returns [`PreprocessError::Io`] on write failure.
pub fn write_tokens_be16(path: impl AsRef<Path>, tokens: &[u16]) -> Result<(), PreprocessError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(256 * 1024, file);
    for &tok in tokens {
        writer.write_all(&tok.to_be_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

/// Reads big-endian `u16` token IDs from `path`.
///
/// # Errors
///
/// Returns [`PreprocessError::OddTokenStreamLength`] or [`PreprocessError::Io`].
pub fn read_tokens_be16(path: impl AsRef<Path>) -> Result<Vec<u16>, PreprocessError> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 2 != 0 {
        return Err(PreprocessError::OddTokenStreamLength(bytes.len() as u64));
    }
    let mut tokens = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        tokens.push(u16::from_be_bytes([chunk[0], chunk[1]]));
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn sample_corpus() -> Vec<u8> {
        let block = concat!(
            "<page>\n",
            "  <title>Wikipedia: Neural Data Compression &amp; Rust</title>\n",
            "  <text xml:space=\"preserve\">\n",
            "    The QUICK brown fox jumps over the lazy dog.\n",
            "    Backslashes \\ and newlines\nappear alongside UTF-8: café, naïve, 日本語, —!\n",
            "    Numbers 1234567890, URLs https://example.org/wiki/Compression, and repetition.\n",
            "  </text>\n",
            "</page>\n",
        );
        block.repeat(12).into_bytes()
    }

    #[test]
    fn encode_decode_roundtrip_discovery_and_frequency() {
        let corpus = sample_corpus();
        for order in [SortOrder::Discovery, SortOrder::ByFrequency] {
            let debug_tmp = tempfile::NamedTempFile::new().unwrap();
            let cfg = EncodeConfig {
                max_words: 400,
                min_freq: 3,
                sort_order: order,
                debug_dict_path: Some(debug_tmp.path().to_path_buf()),
            };
            let (dict, tokens, report) = encode_bytes(&corpus, &cfg).unwrap();
            assert!(dict.len() >= 256);
            assert!(dict.len() <= 400);
            assert_eq!(report.token_count, tokens.len());
            assert!(tokens.len() < corpus.len());

            let decoded = decode_tokens(&tokens, &dict).unwrap();
            assert_eq!(decoded, corpus, "roundtrip failed for {order:?}");

            let debug_bytes = std::fs::read(debug_tmp.path()).unwrap();
            assert!(debug_bytes.windows(4).any(|w| w == b"CUM%"));
        }
    }

    #[test]
    fn empty_input_roundtrip() {
        let (dict, tokens, report) = encode_bytes(b"", &EncodeConfig::default()).unwrap();
        assert_eq!(dict.len(), 256);
        assert!(tokens.is_empty());
        assert_eq!(report.token_count, 0);
        assert_eq!(decode_tokens(&tokens, &dict).unwrap(), b"");
    }

    #[test]
    fn encode_and_decode_file_with_enwik9_voc() {
        let voc_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/enwik9.voc");
        let dict = Dictionary::load(&voc_path).unwrap();
        let corpus = sample_corpus();
        let tokens = encode_with_dictionary(&corpus, &dict).unwrap();
        assert!(tokens.len() < corpus.len());

        let dir = tempfile::tempdir().unwrap();
        let pp_path = dir.path().join("sample.pp");
        let out_path = dir.path().join("sample.out");
        write_tokens_be16(&pp_path, &tokens).unwrap();
        let read_back = read_tokens_be16(&pp_path).unwrap();
        assert_eq!(read_back, tokens);

        let decoded_count = decode_file(&pp_path, &out_path, &voc_path).unwrap();
        assert_eq!(decoded_count, tokens.len());
        assert_eq!(std::fs::read(&out_path).unwrap(), corpus);
    }

    #[test]
    fn decode_rejects_invalid_symbol_and_odd_length() {
        let dict = Dictionary::from_words((0u8..=255).map(|b| vec![b]).collect());
        assert!(matches!(
            decode_tokens(&[300], &dict),
            Err(PreprocessError::InvalidSymbol {
                symbol: 300,
                vocab_size: 256
            })
        ));

        let dir = tempfile::tempdir().unwrap();
        let voc_path = dir.path().join("tiny.voc");
        let pp_path = dir.path().join("odd.pp");
        let out_path = dir.path().join("out.bin");
        dict.save(&voc_path).unwrap();
        std::fs::write(&pp_path, [0, 1, 2]).unwrap();
        assert!(matches!(
            decode_file(&pp_path, &out_path, &voc_path),
            Err(PreprocessError::OddTokenStreamLength(3))
        ));
    }

    #[test]
    fn encode_terminates_on_oscillating_merge_cycle() {
        let crates_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let mut rs_paths = Vec::new();
        for entry in std::fs::read_dir(&crates_dir).unwrap() {
            let src_dir = entry.unwrap().path().join("src");
            if src_dir.is_dir() {
                for src_entry in std::fs::read_dir(src_dir).unwrap() {
                    let p = src_entry.unwrap().path();
                    if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                        rs_paths.push(p);
                    }
                }
            }
        }
        rs_paths.sort();
        let mut corpus = Vec::new();
        for (idx, p) in rs_paths.iter().enumerate() {
            if idx > 0 {
                corpus.push(b'\n');
            }
            corpus.extend_from_slice(&std::fs::read(p).unwrap());
            if corpus.len() >= 32_768 {
                break;
            }
        }
        corpus.truncate(32_768);

        let cfg = EncodeConfig {
            max_words: 1024,
            min_freq: 3,
            sort_order: SortOrder::Discovery,
            debug_dict_path: None,
        };
        let (dict, tokens, _report) = encode_bytes(&corpus, &cfg).unwrap();
        let decoded = decode_tokens(&tokens, &dict).unwrap();
        assert_eq!(decoded, corpus);
    }
}

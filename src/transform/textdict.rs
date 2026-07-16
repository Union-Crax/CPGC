//! Adaptive word-dictionary text transform (WRT-style).
//!
//! On natural-language text, most bytes belong to words drawn from a small
//! vocabulary. Replacing each frequent word with a 1-2 byte token both
//! shortens the stream and — more importantly for a context-mixing coder —
//! lets fixed-order contexts span several *words* instead of several
//! characters, which is worth far more than the raw shrinkage.
//!
//! Unlike classic WRT this needs no shipped dictionary: the vocabulary is
//! mined from the input itself and stored (once, compressibly) at the front
//! of the transformed stream, so the transform is fully self-contained,
//! language-agnostic, and exactly invertible.
//!
//! Token safety: tokens are built exclusively from byte values that do not
//! occur anywhere in the input (there are typically 60+ such bytes in UTF-8
//! text), so no escaping is ever needed. Single-byte tokens go to the
//! highest-savings words; the rest get (lead, index) pairs. A "word" is a
//! maximal alphanumeric run, optionally including one preceding space —
//! folding the near-universal space-before-word into the token roughly
//! doubles the effective win.
//!
//! Serialized layout of a transformed stream:
//! ```text
//! [0]        format: 0x01
//! [1]        n_single: u8
//! [2]        n_lead: u8
//! [3..]      n_single bytes: single-token byte values (dict entries 0..n_single)
//! [..]       n_lead bytes: lead byte values
//! [..+4]     n_words: u32 LE
//! per word:  len: u8, then `len` bytes
//! [rest]     tokenized stream
//! ```
//! Dict entry `i < n_single` is spelled by single byte `singles[i]`; entry
//! `i >= n_single` is spelled `leads[(i - n_single) / 256]` followed by
//! `(i - n_single) % 256`.

use std::collections::HashMap;

const FORMAT: u8 = 0x01;
const MAX_WORD: usize = 40; // cap on dictionary entry length (incl. space)
const MIN_WORD: usize = 3; // shorter entries can't save with a 2-byte token
const MAX_SINGLE: usize = 24; // unused bytes spent on 1-byte tokens
const MAX_LEAD: usize = 40; // unused bytes spent as 2-byte token leads
const MIN_COUNT: u32 = 8; // a word must repeat this often to be considered

/// Quick check that the input looks like text: mostly ASCII letters, digits,
/// whitespace and punctuation. Non-text inputs skip the transform outright.
fn looks_texty(data: &[u8]) -> bool {
    if data.len() < 1 << 16 {
        return false;
    }
    // Sample up to ~1 MB spread across the input.
    let step = (data.len() / (1 << 20)).max(1);
    let mut textish = 0usize;
    let mut total = 0usize;
    let mut i = 0;
    while i < data.len() {
        let b = data[i];
        if b.is_ascii_alphanumeric() || b == b' ' || b == b'\n' || b.is_ascii_punctuation() {
            textish += 1;
        }
        total += 1;
        i += step;
    }
    textish * 100 >= total * 70
}

#[inline]
fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric()
}

/// Scan the maximal candidate run starting at `i`: an optional single space
/// followed by an alphanumeric run. Returns the end index (exclusive), or
/// `i` if there is no candidate here.
#[inline]
fn run_end(data: &[u8], i: usize) -> usize {
    let mut j = i;
    if data[j] == b' ' {
        j += 1;
        if j >= data.len() || !is_word_byte(data[j]) {
            return i;
        }
    } else if !is_word_byte(data[j]) {
        return i;
    }
    while j < data.len() && is_word_byte(data[j]) {
        j += 1;
    }
    j
}

/// Apply the transform. Returns `None` when the input is not texty, has too
/// few unused byte values, or the transform doesn't actually shrink it.
pub fn apply(data: &[u8]) -> Option<Vec<u8>> {
    if !looks_texty(data) {
        return None;
    }

    // ------------------------------------------------------------------
    // Pass 1: byte histogram (for the unused-byte token alphabet).
    // ------------------------------------------------------------------
    let mut used = [false; 256];
    for &b in data {
        used[b as usize] = true;
    }
    let unused: Vec<u8> = (0u8..=255).filter(|&b| !used[b as usize]).collect();
    if unused.len() < 12 {
        return None; // not enough headroom for a useful token alphabet
    }

    // ------------------------------------------------------------------
    // Pass 2: count candidate words (space-prefixed runs counted as spelled).
    // ------------------------------------------------------------------
    // Keyed by the run's bytes; values: (count, first occurrence start/len).
    let mut counts: HashMap<&[u8], u32> = HashMap::with_capacity(1 << 20);
    let mut i = 0usize;
    while i < data.len() {
        let j = run_end(data, i);
        if j > i {
            let run = &data[i..j];
            if run.len() >= MIN_WORD && run.len() <= MAX_WORD {
                *counts.entry(run).or_insert(0) += 1;
            }
            i = j;
        } else {
            i += 1;
        }
    }

    // ------------------------------------------------------------------
    // Select the dictionary: rank by saved bytes assuming a 2-byte token.
    // ------------------------------------------------------------------
    let n_single = MAX_SINGLE.min(unused.len().saturating_sub(MAX_LEAD).max(unused.len() / 3));
    let n_lead = (unused.len() - n_single).min(MAX_LEAD);
    let capacity = n_single + n_lead * 256;

    let mut ranked: Vec<(&[u8], u32)> = counts
        .into_iter()
        .filter(|&(w, c)| c >= MIN_COUNT && (w.len() as u32 - 2) * c > 64)
        .collect();
    if ranked.len() < 16 {
        return None; // not enough repetition to be worth a dictionary
    }
    ranked.sort_unstable_by_key(|&(w, c)| std::cmp::Reverse((w.len() as u64 - 2) * c as u64));
    ranked.truncate(capacity);
    // The single-byte tokens go to the words with the highest 1-extra-byte
    // saving (they already lead the sort in practice; re-rank to be exact).
    let head = ranked.len().min(n_single * 4);
    ranked[..head]
        .sort_unstable_by_key(|&(w, c)| std::cmp::Reverse((w.len() as u64 - 1) * c as u64));

    // Single-token words keep savings order; pair-token words are sorted
    // alphabetically so that similar words share a lead byte — the coder's
    // order-1/2 contexts then see word-class structure instead of noise.
    let n_single_words = ranked.len().min(n_single);
    ranked[n_single_words..].sort_unstable_by_key(|&(w, _)| w);
    let words: Vec<&[u8]> = ranked.iter().map(|&(w, _)| w).collect();
    let index: HashMap<&[u8], usize> =
        words.iter().enumerate().map(|(k, &w)| (w, k)).collect();

    let singles = &unused[..n_single];
    let leads = &unused[n_single..n_single + n_lead];

    // ------------------------------------------------------------------
    // Serialize the dictionary header.
    // ------------------------------------------------------------------
    let mut out = Vec::with_capacity(data.len() / 2 + (1 << 20));
    out.push(FORMAT);
    out.push(n_single as u8);
    out.push(n_lead as u8);
    out.extend_from_slice(singles);
    out.extend_from_slice(leads);
    out.extend_from_slice(&(words.len() as u32).to_le_bytes());
    for w in &words {
        out.push(w.len() as u8);
        out.extend_from_slice(w);
    }

    // ------------------------------------------------------------------
    // Pass 3: tokenize.
    // ------------------------------------------------------------------
    let emit = |out: &mut Vec<u8>, idx: usize| {
        if idx < n_single {
            out.push(singles[idx]);
        } else {
            let j = idx - n_single;
            out.push(leads[j >> 8]);
            out.push((j & 0xff) as u8);
        }
    };

    let mut i = 0usize;
    while i < data.len() {
        let j = run_end(data, i);
        if j > i {
            let run = &data[i..j];
            if run.len() <= MAX_WORD {
                if let Some(&idx) = index.get(run) {
                    emit(&mut out, idx);
                    i = j;
                    continue;
                }
                // A spaced run that missed may still contain a known bare word.
                if run[0] == b' ' {
                    if let Some(&idx) = index.get(&run[1..]) {
                        out.push(b' ');
                        emit(&mut out, idx);
                        i = j;
                        continue;
                    }
                }
            }
            out.extend_from_slice(run);
            i = j;
        } else {
            out.push(data[i]);
            i += 1;
        }
    }

    // Only worth it if the stream (dictionary included) actually shrank
    // meaningfully — the coder sees a denser alphabet either way.
    if out.len() + (out.len() >> 5) < data.len() {
        Some(out)
    } else {
        None
    }
}

/// Exactly invert [`apply`].
pub fn invert(t: &[u8]) -> Option<Vec<u8>> {
    if t.len() < 7 || t[0] != FORMAT {
        return None;
    }
    let n_single = t[1] as usize;
    let n_lead = t[2] as usize;
    let mut p = 3usize;
    if t.len() < p + n_single + n_lead + 4 {
        return None;
    }
    let singles = &t[p..p + n_single];
    p += n_single;
    let leads = &t[p..p + n_lead];
    p += n_lead;
    let n_words = u32::from_le_bytes(t[p..p + 4].try_into().ok()?) as usize;
    p += 4;

    let mut words: Vec<&[u8]> = Vec::with_capacity(n_words);
    for _ in 0..n_words {
        if p >= t.len() {
            return None;
        }
        let len = t[p] as usize;
        p += 1;
        if p + len > t.len() {
            return None;
        }
        words.push(&t[p..p + len]);
        p += len;
    }

    let mut single_map = [usize::MAX; 256];
    for (k, &b) in singles.iter().enumerate() {
        single_map[b as usize] = k;
    }
    let mut lead_map = [usize::MAX; 256];
    for (k, &b) in leads.iter().enumerate() {
        lead_map[b as usize] = k;
    }

    let mut out = Vec::with_capacity(t.len() * 2);
    while p < t.len() {
        let b = t[p] as usize;
        if single_map[b] != usize::MAX {
            out.extend_from_slice(words.get(single_map[b])?);
            p += 1;
        } else if lead_map[b] != usize::MAX {
            if p + 1 >= t.len() {
                return None;
            }
            let idx = n_single + (lead_map[b] << 8) + t[p + 1] as usize;
            out.extend_from_slice(words.get(idx)?);
            p += 2;
        } else {
            out.push(t[p]);
            p += 1;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_text(n: usize) -> Vec<u8> {
        let para = "The quick brown fox jumps over the lazy dog. In 2005 the \
                    encyclopedia article described [[compression]] algorithms and \
                    the information theory behind them, according to the text.\n";
        para.as_bytes().iter().cycle().take(n).cloned().collect()
    }

    #[test]
    fn roundtrip_text() {
        let data = sample_text(1 << 17);
        let t = apply(&data).expect("texty input should transform");
        assert!(t.len() < data.len(), "transform must shrink text");
        let back = invert(&t).expect("invert failed");
        assert_eq!(back, data);
    }

    #[test]
    fn skips_binary() {
        let mut x: u64 = 99;
        let data: Vec<u8> = (0..1 << 17)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                (x >> 56) as u8
            })
            .collect();
        assert!(apply(&data).is_none(), "random bytes must not transform");
    }

    #[test]
    fn skips_small() {
        assert!(apply(b"tiny input").is_none());
    }

    #[test]
    fn roundtrip_with_all_byte_values_in_words() {
        // Words adjacent to every printable byte; transform may or may not
        // fire, but if it does the roundtrip must hold.
        let mut data = sample_text(1 << 17);
        data.extend((0u8..=255).cycle().take(4096));
        if let Some(t) = apply(&data) {
            assert_eq!(invert(&t).unwrap(), data);
        }
    }
}

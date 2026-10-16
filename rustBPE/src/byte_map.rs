const fn is_visible(b: u8) -> bool {
    matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF)
}

const fn build_byte_to_char() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next = 256u32;
    let mut i = 0;

    while i < 256 {
        let b = i as u8;
        table[i] = if is_visible(b) {
            b as char
        } else {
            let c = char::from_u32(next).unwrap();
            next += 1;
            c
        };
        i += 1;
    }
    table
}

pub const BYTE_TO_CHAR: [char; 256] = build_byte_to_char();

pub fn visualize(byte: &[u8]) -> String {
    byte.iter().map(|&b| BYTE_TO_CHAR[b as usize]).collect()
}

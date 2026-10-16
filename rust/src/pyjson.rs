//! JSON text formatted exactly like Python's ``json.dumps``.
//!
//! Output files (metrics, manifests, identities) keep the byte layout of the
//! Python implementation: ``", "``/``": "`` separators, Python float ``repr``
//! and optional key sorting and indentation.

use serde_json::Value;

use crate::frames::python_float_repr;

#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub sort_keys: bool,
    pub indent: Option<usize>,
    pub ensure_ascii: bool,
}

impl Options {
    pub const COMPACT: Options = Options {
        sort_keys: false,
        indent: None,
        ensure_ascii: true,
    };
}

pub fn dumps(value: &Value, options: Options) -> String {
    let mut out = String::new();
    write_value(&mut out, value, options, 0);
    out
}

fn write_string(out: &mut String, text: &str, ensure_ascii: bool) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ensure_ascii && (c as u32) > 0x7e => {
                let mut buffer = [0u16; 2];
                for unit in c.encode_utf16(&mut buffer) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_number(out: &mut String, number: &serde_json::Number) {
    if let Some(i) = number.as_i64() {
        out.push_str(&i.to_string());
    } else if let Some(u) = number.as_u64() {
        out.push_str(&u.to_string());
    } else if let Some(f) = number.as_f64() {
        out.push_str(&python_float_repr(f));
    }
}

fn newline(out: &mut String, options: Options, level: usize) {
    if let Some(indent) = options.indent {
        out.push('\n');
        out.push_str(&" ".repeat(indent * level));
    }
}

fn write_value(out: &mut String, value: &Value, options: Options, level: usize) {
    let item_separator = if options.indent.is_some() { "," } else { ", " };
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => write_number(out, n),
        Value::String(s) => write_string(out, s, options.ensure_ascii),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_separator);
                }
                newline(out, options, level + 1);
                write_value(out, item, options, level + 1);
            }
            newline(out, options, level);
            out.push(']');
        }
        Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            if options.sort_keys {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            out.push('{');
            for (i, (key, item)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(item_separator);
                }
                newline(out, options, level + 1);
                write_string(out, key, options.ensure_ascii);
                out.push_str(": ");
                write_value(out, item, options, level + 1);
            }
            newline(out, options, level);
            out.push('}');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn python_layout() {
        let value = json!({"b": 1e-5, "a": [1, 2.5], "c": "é"});
        assert_eq!(
            dumps(
                &value,
                Options {
                    sort_keys: true,
                    indent: None,
                    ensure_ascii: true
                }
            ),
            concat!(r#"{"a": [1, 2.5], "b": 1e-05, "c": ""#, "\\", r#"u00e9"}"#)
        );
        assert_eq!(
            dumps(
                &json!({"a": 1}),
                Options {
                    sort_keys: false,
                    indent: Some(2),
                    ensure_ascii: false
                }
            ),
            "{\n  \"a\": 1\n}"
        );
    }
}

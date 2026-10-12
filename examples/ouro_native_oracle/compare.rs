use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::checkpoint::{DIGEST, REVISION};

pub const ABSOLUTE: f64 = 0.003;
pub const RELATIVE: f64 = 0.003;

pub struct JsonLines {
    writer: BufWriter<File>,
    pub path: PathBuf,
    pub line: usize,
}

impl JsonLines {
    pub fn create(path: &Path) -> io::Result<Self> {
        Ok(Self {
            writer: BufWriter::with_capacity(1024 * 1024, File::create_new(path)?),
            path: path.to_path_buf(),
            line: 0,
        })
    }

    pub fn emit(&mut self, record: &Value) -> io::Result<usize> {
        serde_json::to_writer(&mut self.writer, record).map_err(io::Error::other)?;
        self.writer.write_all(b"\n")?;
        self.line += 1;
        Ok(self.line)
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

#[derive(Clone)]
pub struct Vector {
    pub values: Vec<f32>,
    pub shape: Vec<usize>,
    pub path: PathBuf,
    pub line: usize,
    pub field: String,
}

pub struct Archive {
    pub vectors: BTreeMap<String, Vector>,
    pub context: Value,
}

pub fn read_values(record: &Value, field: &str) -> io::Result<Vec<f32>> {
    record[field]
        .as_array()
        .ok_or_else(|| io::Error::other(format!("vector field {field} absent")))?
        .iter()
        .map(|value| {
            let value = value
                .as_f64()
                .ok_or_else(|| io::Error::other("vector coordinate is not a number"))?
                as f32;
            if value.is_finite() {
                Ok(value)
            } else {
                Err(io::Error::other("vector coordinate is nonfinite"))
            }
        })
        .collect()
}

pub fn visit_records(
    path: &Path,
    mut callback: impl FnMut(usize, Value) -> io::Result<()>,
) -> io::Result<()> {
    let reader = BufReader::new(File::open(path)?);
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            return Err(io::Error::other(format!(
                "{}:{} empty JSONL record",
                path.display(),
                index + 1
            )));
        }
        let value: Value = serde_json::from_str(&line).map_err(io::Error::other)?;
        callback(index + 1, value)?;
    }
    Ok(())
}

pub fn layer_key(iteration: usize, layer: usize, boundary: &str) -> String {
    format!("loop{iteration}/layer{layer}/{boundary}")
}

pub fn loop_key(iteration: usize, boundary: &str) -> String {
    format!("loop{iteration}/{boundary}")
}

fn index_field(record: &Value, field: &str) -> io::Result<usize> {
    record[field]
        .as_u64()
        .ok_or_else(|| io::Error::other(format!("unsigned {field} absent")))
        .and_then(|number| usize::try_from(number).map_err(io::Error::other))
}

impl Archive {
    pub fn open(path: &Path, prefix: &[u32]) -> io::Result<Self> {
        let mut vectors = BTreeMap::new();
        let mut context = Value::Null;
        visit_records(path, |line, record| {
            if record["context"]["arm"] != "baseline4" {
                return Ok(());
            }
            let candidate = &record["context"];
            let policy = candidate["numeric_policy"].as_str().unwrap_or_default();
            if candidate["prefix_ids"] != json!(prefix)
                || candidate["backend"] != "cpu"
                || candidate["checkpoint_revision"] != REVISION
                || candidate["checkpoint_expected_sha256"] != DIGEST
                || candidate["checkpoint_observed_sha256"] != DIGEST
                || candidate["depth"] != 4
                || candidate["use_cache"] != false
                || candidate["readout"] != "explicit terminal recurrent step"
                || !policy.contains("f32 activation arithmetic")
                || !policy.contains("author BF16 intermediate rounding not reproduced")
            {
                return Err(io::Error::other(format!(
                    "{}:{line} archive conditions differ",
                    path.display()
                )));
            }
            if context.is_null() {
                context = candidate.clone();
            }
            let mut insert = |key: String, field: &str, shape: Vec<usize>| -> io::Result<()> {
                let values = read_values(&record, field)?;
                if values.len() != shape.iter().product::<usize>() {
                    return Err(io::Error::other(format!("{key}: coordinate count differs")));
                }
                if vectors
                    .insert(
                        key.clone(),
                        Vector {
                            values,
                            shape,
                            path: path.to_path_buf(),
                            line,
                            field: field.to_owned(),
                        },
                    )
                    .is_some()
                {
                    return Err(io::Error::other(format!(
                        "duplicate archive boundary {key}"
                    )));
                }
                Ok(())
            };
            match record["kind"].as_str() {
                Some("layer_execution") => {
                    let iteration = index_field(&record, "iteration")?;
                    let layer = index_field(&record, "physical_layer")?;
                    if !(1..=4).contains(&iteration) || layer >= 24 {
                        return Err(io::Error::other("archive layer index outside expected set"));
                    }
                    for (boundary, field) in
                        [("input", "input_hidden"), ("output", "output_hidden")]
                    {
                        insert(
                            layer_key(iteration, layer, boundary),
                            field,
                            vec![prefix.len(), 2048],
                        )?;
                    }
                }
                Some("loop_boundary") => {
                    let iteration = index_field(&record, "iteration")?;
                    if !(1..=4).contains(&iteration)
                        || record["next_loop_input"] != "normalized_hidden"
                        || record["normalization_tensor"] != "model.norm.weight"
                    {
                        return Err(io::Error::other("archive recurrence boundary differs"));
                    }
                    insert(
                        loop_key(iteration, "pre_norm"),
                        "pre_norm_hidden",
                        vec![prefix.len(), 2048],
                    )?;
                    insert(
                        loop_key(iteration, "post_norm"),
                        "normalized_hidden",
                        vec![prefix.len(), 2048],
                    )?;
                }
                Some("diagnostic_readouts") => {
                    insert(
                        "reference_logits".to_owned(),
                        "reference_logits",
                        vec![49152],
                    )?;
                    insert("terminal_logits".to_owned(), "terminal_logits", vec![49152])?;
                }
                _ => {}
            }
            Ok(())
        })?;
        let archive = Self { vectors, context };
        archive.require_complete()?;
        Ok(archive)
    }

    pub fn require_complete(&self) -> io::Result<()> {
        if self.vectors.len() != 202 {
            return Err(io::Error::other(format!(
                "archive vectors={}, expected=202",
                self.vectors.len()
            )));
        }
        for iteration in 1..=4 {
            for layer in 0..24 {
                self.get(&layer_key(iteration, layer, "input"))?;
                self.get(&layer_key(iteration, layer, "output"))?;
            }
            self.get(&loop_key(iteration, "pre_norm"))?;
            self.get(&loop_key(iteration, "post_norm"))?;
        }
        self.get("reference_logits")?;
        self.get("terminal_logits")?;
        Ok(())
    }

    pub fn get(&self, key: &str) -> io::Result<&Vector> {
        self.vectors
            .get(key)
            .ok_or_else(|| io::Error::other(format!("required boundary {key} absent")))
    }

    pub fn oracle(directory: &Path) -> io::Result<Self> {
        let path = directory.join("vectors.jsonl");
        let mut vectors = BTreeMap::new();
        let context: Value = serde_json::from_slice(&fs::read(directory.join("context.json"))?)
            .map_err(io::Error::other)?;
        visit_records(&path, |line, record| {
            let key = record["key"]
                .as_str()
                .ok_or_else(|| io::Error::other("oracle vector key absent"))?;
            let shape: Vec<usize> =
                serde_json::from_value(record["shape"].clone()).map_err(io::Error::other)?;
            let values = read_values(&record, "values")?;
            if values.len() != shape.iter().product::<usize>() {
                return Err(io::Error::other("oracle shape differs"));
            }
            if vectors
                .insert(
                    key.to_owned(),
                    Vector {
                        values,
                        shape,
                        path: path.clone(),
                        line,
                        field: "values".to_owned(),
                    },
                )
                .is_some()
            {
                return Err(io::Error::other("oracle vector key duplicate"));
            }
            Ok(())
        })?;
        Ok(Self { vectors, context })
    }
}

#[derive(Default, Clone)]
pub struct Counts {
    pub compared: usize,
    pub exact: usize,
    pub exceeding: usize,
    pub positive: usize,
    pub negative: usize,
    pub zero_bits: usize,
    pub maximum_absolute: f64,
    pub maximum_index: usize,
}

pub fn counts(
    actual: &[f32],
    reference: &[f32],
    absolute: f64,
    relative: f64,
) -> io::Result<Counts> {
    if actual.len() != reference.len() {
        return Err(io::Error::other("comparison vector dimensions differ"));
    }
    let mut result = Counts::default();
    for (index, (actual_value, reference_value)) in actual.iter().zip(reference).enumerate() {
        if !actual_value.is_finite() || !reference_value.is_finite() {
            return Err(io::Error::other("comparison coordinate nonfinite"));
        }
        let difference = f64::from(*actual_value) - f64::from(*reference_value);
        let threshold =
            absolute + relative * f64::from(actual_value.abs().max(reference_value.abs()));
        result.compared += 1;
        if actual_value.to_bits() == reference_value.to_bits() {
            result.exact += 1;
        } else if difference > 0.0 {
            result.positive += 1;
        } else if difference < 0.0 {
            result.negative += 1;
        } else {
            result.zero_bits += 1;
        }
        if difference.abs() > threshold {
            result.exceeding += 1;
        }
        if difference.abs() > result.maximum_absolute {
            result.maximum_absolute = difference.abs();
            result.maximum_index = index;
        }
    }
    Ok(result)
}

pub struct Comparisons {
    pub vectors: JsonLines,
    pub differences: JsonLines,
    pub count: usize,
    pub coordinates: usize,
    pub non_exact: usize,
    pub exceeding: usize,
}

impl Comparisons {
    pub fn create(directory: &Path) -> io::Result<Self> {
        Ok(Self {
            vectors: JsonLines::create(&directory.join("comparisons.jsonl"))?,
            differences: JsonLines::create(&directory.join("differences.jsonl"))?,
            count: 0,
            coordinates: 0,
            non_exact: 0,
            exceeding: 0,
        })
    }

    pub fn compare(
        &mut self,
        key: &str,
        actual: &Vector,
        reference: &Vector,
        condition: &str,
    ) -> io::Result<()> {
        if actual.shape != reference.shape {
            return Err(io::Error::other(format!("{key}: vector shapes differ")));
        }
        let result = counts(&actual.values, &reference.values, ABSOLUTE, RELATIVE)?;
        let identifier = self.count;
        let mut stride = Vec::with_capacity(actual.shape.len());
        for dimension in 0..actual.shape.len() {
            stride.push(actual.shape[dimension + 1..].iter().product::<usize>());
        }
        for (index, (actual_value, reference_value)) in
            actual.values.iter().zip(&reference.values).enumerate()
        {
            if actual_value.to_bits() == reference_value.to_bits() {
                continue;
            }
            let difference = f64::from(*actual_value) - f64::from(*reference_value);
            let threshold =
                ABSOLUTE + RELATIVE * f64::from(actual_value.abs().max(reference_value.abs()));
            let coordinates: Vec<usize> = stride
                .iter()
                .zip(&actual.shape)
                .map(|(stride, extent)| (index / stride) % extent)
                .collect();
            let direction = if difference > 0.0 {
                "positive"
            } else if difference < 0.0 {
                "negative"
            } else {
                "zero_difference_bits"
            };
            self.differences.emit(&json!({"kind":"coordinate_difference","comparison":identifier,"index":index,"coordinates":coordinates,"reference":reference_value,"actual":actual_value,"reference_bits":reference_value.to_bits(),"actual_bits":actual_value.to_bits(),"difference":difference,"absolute_error":difference.abs(),"threshold":threshold,"exceeding":difference.abs()>threshold,"direction":direction}))?;
        }
        self.vectors.emit(&json!({"kind":"vector_comparison","id":identifier,"key":key,"condition":condition,"shape":actual.shape,"coordinates":result.compared,"absolute_tolerance":ABSOLUTE,"relative_tolerance":RELATIVE,"exact_equal":result.exact,"within_tolerance":result.compared-result.exceeding,"exceeding":result.exceeding,"positive":result.positive,"negative":result.negative,"zero_difference_bits":result.zero_bits,"nonfinite":0,"maximum_absolute_error":result.maximum_absolute,"maximum_index":result.maximum_index,"actual":{"path":actual.path,"line":actual.line,"field":actual.field},"reference":{"path":reference.path,"line":reference.line,"field":reference.field}}))?;
        self.count += 1;
        self.coordinates += result.compared;
        self.non_exact += result.compared - result.exact;
        self.exceeding += result.exceeding;
        Ok(())
    }

    pub fn finish(&mut self) -> io::Result<Value> {
        self.vectors.flush()?;
        self.differences.flush()?;
        Ok(
            json!({"comparison_records":self.count,"coordinate_comparisons":self.coordinates,"non_exact_coordinates":self.non_exact,"difference_rows":self.differences.line,"exceeding_coordinates":self.exceeding,"structural_errors":0,"nonfinite_coordinates":0,"absolute_tolerance":ABSOLUTE,"relative_tolerance":RELATIVE}),
        )
    }
}

pub fn vector(
    writer: &mut JsonLines,
    key: &str,
    values: &[f32],
    shape: &[usize],
    mode: &str,
) -> io::Result<Vector> {
    if values.len() != shape.iter().product::<usize>()
        || values.iter().any(|value| !value.is_finite())
    {
        return Err(io::Error::other(
            "oracle vector shape or finite condition differs",
        ));
    }
    let line = writer.emit(
        &json!({"kind":"oracle_vector","key":key,"shape":shape,"mode":mode,"values":values}),
    )?;
    Ok(Vector {
        values: values.to_vec(),
        shape: shape.to_vec(),
        path: writer.path.clone(),
        line,
        field: "values".to_owned(),
    })
}

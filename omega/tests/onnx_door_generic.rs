//! The ONNX door is generic: a convolutional classifier of another family than the embedding model the
//! parity cells use runs through the same `OnnxDoor::from_bytes` on CPU and Metal, and the door sources
//! name no model. Top-1 is scored over the first 1000 images of the MNIST `t10k` split.

#![cfg(feature = "onnx-door")]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, ensure};
use omega::onnx_door::{DoorBatch, DoorConfig, OnnxDoor};
use proxima_primitives::pipe::Pipe;

const MODEL_PATH_ENV: &str = "MNIST_MODEL";
const MODEL_DEFAULT_PATH: &str =
    "/Users/brianbruggeman/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx";
const DATASET_RELATIVE_DIRECTORY: &str = ".cache/burn-dataset/mnist/test";
const IMAGES_FILE: &str = "t10k-images-idx3-ubyte";
const LABELS_FILE: &str = "t10k-labels-idx1-ubyte";
const INPUT_NAME: &str = "input.1";
const IMAGE_COUNT: usize = 1000;
const CLASS_COUNT: u64 = 10;
const MIN_ACCURACY: f64 = 0.98;
const PIXEL_MEAN: f32 = 0.1307;
const PIXEL_DEVIATION: f32 = 0.3081;
const MODEL_NAMES: [&str; 3] = ["bge", "bert", "mnist"];
const DOOR_SOURCE_DIRECTORIES: [&str; 2] = ["src/onnx_door", "src/door"];

#[derive(Clone, Copy)]
enum Target {
    Cpu,
    #[cfg(feature = "metal")]
    Metal,
}

impl Target {
    fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            #[cfg(feature = "metal")]
            Self::Metal => "metal",
        }
    }

    fn config(self) -> DoorConfig {
        match self {
            Self::Cpu => DoorConfig::default(),
            #[cfg(feature = "metal")]
            Self::Metal => DoorConfig::builder()
                .engine("gpu")
                .gpu_driver("metal")
                .build(),
        }
    }
}

fn model_path() -> PathBuf {
    env::var_os(MODEL_PATH_ENV).map_or_else(|| PathBuf::from(MODEL_DEFAULT_PATH), PathBuf::from)
}

fn read_dataset_file(name: &str) -> Result<Vec<u8>> {
    let path = PathBuf::from(env::var_os("HOME").context("HOME is not set")?)
        .join(DATASET_RELATIVE_DIRECTORY)
        .join(name);
    fs::read(&path).with_context(|| format!("read mnist dataset file {}", path.display()))
}

fn be_u32(bytes: &[u8], offset: usize) -> Result<usize> {
    let word: [u8; 4] = bytes
        .get(offset..offset + 4)
        .context("idx header is truncated")?
        .try_into()?;
    Ok(usize::try_from(u32::from_be_bytes(word))?)
}

fn idx_layout(bytes: &[u8]) -> Result<(usize, usize, usize)> {
    let dimensions = usize::from(*bytes.get(3).context("idx header is truncated")?);
    let count = be_u32(bytes, 4)?;
    let item_size = (2..=dimensions)
        .map(|axis| be_u32(bytes, axis * 4))
        .product::<Result<usize>>()?;
    ensure!(item_size > 0, "idx items have no extent");
    Ok((count, item_size, 4 + dimensions * 4))
}

fn idx_items(bytes: &[u8], limit: usize) -> Result<Vec<&[u8]>> {
    let (count, item_size, offset) = idx_layout(bytes)?;
    ensure!(
        count >= limit,
        "idx file holds {count} items, expected at least {limit}"
    );
    let payload = bytes.get(offset..).context("idx payload is missing")?;
    ensure!(
        payload.len() >= limit * item_size,
        "idx payload is {} bytes, expected at least {}",
        payload.len(),
        limit * item_size
    );
    Ok(payload.chunks_exact(item_size).take(limit).collect())
}

fn normalized_images(count: usize) -> Result<Vec<Vec<f32>>> {
    let bytes = read_dataset_file(IMAGES_FILE)?;
    Ok(idx_items(&bytes, count)?
        .into_iter()
        .map(|pixels| {
            pixels
                .iter()
                .map(|pixel| (f32::from(*pixel) / 255.0 - PIXEL_MEAN) / PIXEL_DEVIATION)
                .collect()
        })
        .collect())
}

fn labels(count: usize) -> Result<Vec<usize>> {
    let bytes = read_dataset_file(LABELS_FILE)?;
    Ok(idx_items(&bytes, count)?
        .into_iter()
        .map(|item| usize::from(item[0]))
        .collect())
}

fn top1_accuracy(predicted: &[usize], expected: &[usize]) -> f64 {
    let correct = predicted
        .iter()
        .zip(expected)
        .filter(|(left, right)| left == right)
        .count();
    correct as f64 / expected.len() as f64
}

fn argmax(scores: &[f32]) -> Result<usize> {
    scores
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .context("the door returned no class scores")
}

fn build_door(target: Target) -> Result<OnnxDoor> {
    let path = model_path();
    let bytes = fs::read(&path).with_context(|| format!("read model {}", path.display()))?;
    OnnxDoor::from_bytes(bytes, &target.config())
        .map_err(|error| anyhow!("{} door config: {error}", target.label()))
}

async fn classify(door: &OnnxDoor, image: &[f32]) -> Result<usize> {
    let batch = DoorBatch {
        pins: BTreeMap::new(),
        inputs: vec![(INPUT_NAME.to_owned(), image.to_vec())],
    };
    let output = door
        .call(batch)
        .await
        .map_err(|error| anyhow!("door call: {error}"))?;
    let (name, _) = output.outputs.first().context("door has no output")?;
    let (scores, shape) = output.get(name).context("door output is missing")?;
    ensure!(
        shape == [1, CLASS_COUNT],
        "output {name} has shape {shape:?}"
    );
    argmax(scores)
}

async fn predictions(target: Target, images: &[Vec<f32>]) -> Result<Vec<usize>> {
    let door = build_door(target)?;
    let mut predicted = Vec::with_capacity(images.len());
    for image in images {
        predicted.push(classify(&door, image).await?);
    }
    Ok(predicted)
}

async fn top1_through_the_door(target: Target) -> Result<()> {
    let images = normalized_images(IMAGE_COUNT)?;
    let expected = labels(IMAGE_COUNT)?;

    let predicted = predictions(target, &images).await?;

    let accuracy = top1_accuracy(&predicted, &expected);
    println!(
        "onnx_door_generic target={} images={} accuracy={accuracy:.4}",
        target.label(),
        predicted.len()
    );
    ensure!(
        predicted.len() == IMAGE_COUNT,
        "{} images classified",
        predicted.len()
    );
    ensure!(
        accuracy >= MIN_ACCURACY,
        "{} top-1 {accuracy:.4} is below {MIN_ACCURACY}",
        target.label()
    );
    Ok(())
}

fn model_name_count(source: &str) -> usize {
    let lowered = source.to_lowercase();
    MODEL_NAMES
        .iter()
        .map(|name| lowered.matches(name).count())
        .sum()
}

fn rust_sources(relative_directory: &str) -> Result<Vec<PathBuf>> {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative_directory);
    let entries = fs::read_dir(&directory)
        .with_context(|| format!("read door source directory {}", directory.display()))?;
    let paths = entries
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(paths
        .into_iter()
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .collect())
}

#[proxima::test]
async fn onnx_door_generic_mnist_cpu_top1_at_least_0_98_over_first_1000_t10k_images() -> Result<()>
{
    top1_through_the_door(Target::Cpu).await
}

#[cfg(feature = "metal")]
fn metal_bound_bytes() -> u64 {
    omega::metal::BLOCK_NOCOPY_BOUND_BYTES.get() + omega::metal::BLOCK_COPIED_BYTES.get()
}

#[cfg(feature = "metal")]
#[proxima::test]
async fn onnx_door_generic_mnist_metal_top1_at_least_0_98() -> Result<()> {
    let bound_before = metal_bound_bytes();

    top1_through_the_door(Target::Metal).await?;

    let bound = metal_bound_bytes() - bound_before;
    println!("onnx_door_generic target=metal bound_bytes={bound}");
    ensure!(bound > 0, "the metal door bound no bytes to metal buffers");
    Ok(())
}

#[test]
fn onnx_door_generic_labels_compared_with_themselves_score_1_0() -> Result<()> {
    let expected = labels(IMAGE_COUNT)?;

    let accuracy = top1_accuracy(&expected, &expected);

    ensure!(expected.len() == IMAGE_COUNT, "{} labels", expected.len());
    ensure!(
        (accuracy - 1.0).abs() < f64::EPSILON,
        "identity accuracy is {accuracy}"
    );
    Ok(())
}

#[test]
fn onnx_door_generic_labels_shuffled_by_one_position_score_below_0_98() -> Result<()> {
    let expected = labels(IMAGE_COUNT)?;
    let mut shifted = expected.clone();
    shifted.rotate_left(1);

    let accuracy = top1_accuracy(&shifted, &expected);

    println!("onnx_door_generic shifted_labels accuracy={accuracy:.4}");
    ensure!(
        accuracy < MIN_ACCURACY,
        "labels shifted by one position still score {accuracy}"
    );
    Ok(())
}

#[test]
fn onnx_door_generic_door_sources_name_no_model() -> Result<()> {
    for directory in DOOR_SOURCE_DIRECTORIES {
        let sources = rust_sources(directory)?;
        ensure!(!sources.is_empty(), "{directory} holds no rust sources");
        for path in sources {
            let count = model_name_count(&fs::read_to_string(&path)?);
            ensure!(count == 0, "{} names a model {count} times", path.display());
        }
    }
    Ok(())
}

#[test]
fn onnx_door_generic_model_name_counter_control() {
    let counted = model_name_count("the Mnist classifier");

    assert_eq!(counted, 1);
}

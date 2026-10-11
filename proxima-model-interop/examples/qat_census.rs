mod qat;

use std::path::PathBuf;

use qat::QatError;
use qat::graph::{load_descriptor, step_graph};
use qat::report::{print_cell, print_control};

struct Arguments {
    config: PathBuf,
    layers: Vec<u32>,
    sequence: u32,
    control: bool,
}

fn parse_number(text: &str) -> Result<u32, QatError> {
    text.parse()
        .map_err(|_| QatError::Argument(format!("{text} is not an unsigned number")))
}

fn parse_arguments(raw: &[String]) -> Result<Arguments, QatError> {
    let mut arguments = Arguments {
        config: PathBuf::new(),
        layers: Vec::new(),
        sequence: 256,
        control: false,
    };
    for pair in raw.chunks(2) {
        match pair {
            [flag, value] if flag == "--config" => arguments.config = PathBuf::from(value),
            [flag, value] if flag == "--layers" => {
                arguments.layers = value
                    .split(',')
                    .map(parse_number)
                    .collect::<Result<_, _>>()?;
            }
            [flag, value] if flag == "--seq" => arguments.sequence = parse_number(value)?,
            [flag, value] if flag == "--dtype" && value == "f32" => {}
            [flag, value] if flag == "--control" && value == "int32_input" => {
                arguments.control = true;
            }
            other => return Err(QatError::Argument(format!("unrecognized {other:?}"))),
        }
    }
    Ok(arguments)
}

fn main() -> Result<(), QatError> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let arguments = parse_arguments(&raw)?;
    if arguments.control {
        return print_control();
    }
    for layers in &arguments.layers {
        let descriptor = load_descriptor(&arguments.config, *layers)?;
        let graph = step_graph(&descriptor, arguments.sequence)?;
        print_cell(*layers, &graph)?;
    }
    Ok(())
}

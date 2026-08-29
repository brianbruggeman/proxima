/// the run controller: a small state machine that walks the stages in order,
/// then hands off to reporting. transitions are explicit so the driver never
/// has to track where it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Stage(usize),
    Report,
}

pub struct RunController {
    phase: Phase,
    stages: usize,
}

impl RunController {
    pub fn new(stages: usize) -> Self {
        Self { phase: Phase::Idle, stages }
    }

    pub fn advance(&mut self) -> Phase {
        self.phase = match self.phase {
            Phase::Idle => self.first(),
            Phase::Stage(i) => {
                if i + 1 < self.stages {
                    Phase::Stage(i + 1)
                } else {
                    Phase::Report
                }
            }
            Phase::Report => Phase::Report,
        };
        self.phase
    }

    fn first(&self) -> Phase {
        if self.stages == 0 { Phase::Report } else { Phase::Stage(0) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walks_stages_then_reports() {
        let mut c = RunController::new(2);
        assert_eq!(c.advance(), Phase::Stage(0));
        assert_eq!(c.advance(), Phase::Stage(1));
        assert_eq!(c.advance(), Phase::Report);
        assert_eq!(c.advance(), Phase::Report);
    }

    #[test]
    fn no_stages_goes_straight_to_report() {
        let mut c = RunController::new(0);
        assert_eq!(c.advance(), Phase::Report);
    }
}

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

const FEATURES: usize = 7;

#[derive(Clone)]
struct Winner {
    reduced: f64,
    session: usize,
    rank: u32,
    option: usize,
}

impl PartialEq for Winner {
    fn eq(&self, other: &Self) -> bool {
        self.reduced == other.reduced && self.rank == other.rank && self.option == other.option
    }
}

impl Eq for Winner {}

impl PartialOrd for Winner {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Winner {
    fn cmp(&self, other: &Self) -> Ordering {
        self.reduced
            .total_cmp(&other.reduced)
            .then(self.rank.cmp(&other.rank))
            .then(self.option.cmp(&other.option))
    }
}

struct Sweep {
    epoch: u64,
    eta: f64,
    ids: Vec<u64>,
    sessions: Vec<i32>,
    options: Vec<u16>,
    reduced: Vec<f64>,
    costs: Vec<f64>,
    gains: Vec<f64>,
    starts: Vec<i32>,
    rows: Vec<i32>,
    values: Vec<f64>,
    repair: f64,
    minimum: f64,
    violations: u64,
    evaluated: u64,
}

#[pyclass]
struct PricingOracle {
    sessions: usize,
    signatures: usize,
    options: usize,
    resources: usize,
    horizon: f64,
    gains: Vec<f64>,
    features: Vec<f64>,
    feasible: Vec<u16>,
    option_signatures: Vec<u16>,
    option_starts: Vec<i32>,
    resource_rows: Vec<i32>,
    coefficients: Vec<f64>,
    session_ranks: Vec<u32>,
    generated: Vec<u16>,
    loaded: Vec<bool>,
    rank_used: Vec<bool>,
    loaded_count: usize,
    epoch: u64,
    pending: Option<(u64, Vec<u64>)>,
}

fn finite_nonnegative(values: &[f64]) -> bool {
    values
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
}

impl PricingOracle {
    fn value(&self, session: usize, option: usize, entry: usize) -> f64 {
        let signature = self.option_signatures[option] as usize;
        let feature = (session * self.signatures + signature) * FEATURES;
        let coefficient = entry * FEATURES;
        (0..FEATURES)
            .map(|i| self.features[feature + i] * self.coefficients[coefficient + i])
            .sum()
    }

    fn price_inner(
        &mut self,
        phase: u8,
        mut eta: f64,
        resource_duals: &[f64],
        session_duals: &[f64],
        batch: usize,
        tolerance: f64,
    ) -> PyResult<Sweep> {
        if self.pending.is_some() {
            return Err(PyValueError::new_err(
                "previous pricing batch is uncommitted",
            ));
        }
        if self.loaded_count != self.sessions {
            return Err(PyValueError::new_err("pricing SoA is incomplete"));
        }
        if !matches!(phase, 1 | 2)
            || batch == 0
            || !eta.is_finite()
            || eta < 0.0
            || !tolerance.is_finite()
            || tolerance < 0.0
        {
            return Err(PyValueError::new_err("invalid pricing arguments"));
        }
        if resource_duals.len() != self.resources
            || session_duals.len() != self.sessions
            || !finite_nonnegative(resource_duals)
            || !finite_nonnegative(session_duals)
        {
            return Err(PyValueError::new_err("invalid pricing duals"));
        }
        if phase == 1 {
            eta = eta.min(1.0);
        }
        let mut weights = vec![0.0; self.options * FEATURES];
        for option in 0..self.options {
            for entry in
                self.option_starts[option] as usize..self.option_starts[option + 1] as usize
            {
                let dual = resource_duals[self.resource_rows[entry] as usize];
                for feature in 0..FEATURES {
                    weights[option * FEATURES + feature] +=
                        dual * self.coefficients[entry * FEATURES + feature];
                }
            }
            if phase == 2 {
                weights[option * FEATURES + 4] += 1.0 / self.horizon;
            }
            if !weights[option * FEATURES..(option + 1) * FEATURES]
                .iter()
                .all(|value| value.is_finite())
            {
                return Err(PyValueError::new_err("nonfinite pricing weight"));
            }
        }
        let mut heap = BinaryHeap::with_capacity(batch + 1);
        let mut repair = 0.0;
        let mut repair_error = 0.0;
        let mut minimum = f64::INFINITY;
        let mut violations = 0;
        let mut evaluated = 0;
        for session in 0..self.sessions {
            let mut session_minimum = f64::INFINITY;
            let mut best: Option<Winner> = None;
            for option in 0..self.options {
                if self.feasible[session] & (1u16 << option) == 0 {
                    continue;
                }
                evaluated += 1;
                let signature = self.option_signatures[option] as usize;
                let feature = (session * self.signatures + signature) * FEATURES;
                let reduced = (0..FEATURES)
                    .map(|i| self.features[feature + i] * weights[option * FEATURES + i])
                    .sum::<f64>()
                    - eta * self.gains[session]
                    + session_duals[session];
                if !reduced.is_finite() {
                    return Err(PyValueError::new_err("nonfinite reduced cost"));
                }
                session_minimum = session_minimum.min(reduced);
                if self.generated[session] & (1u16 << option) == 0 {
                    let candidate = Winner {
                        reduced,
                        session,
                        rank: self.session_ranks[session],
                        option,
                    };
                    if best.as_ref().map_or(true, |current| candidate < *current) {
                        best = Some(candidate);
                    }
                }
            }
            minimum = minimum.min(session_minimum);
            if session_minimum.is_finite() {
                let correction = (-session_minimum).max(0.0);
                let adjusted = correction - repair_error;
                let next = repair + adjusted;
                repair_error = (next - repair) - adjusted;
                repair = next;
            }
            if let Some(candidate) = best.filter(|candidate| candidate.reduced < -tolerance) {
                violations += 1;
                heap.push(candidate);
                if heap.len() > batch {
                    heap.pop();
                }
            }
        }
        let winners = heap.into_sorted_vec();
        let epoch = self.epoch;
        self.epoch += 1;
        let mut sweep = Sweep {
            epoch,
            eta,
            ids: Vec::with_capacity(winners.len()),
            sessions: Vec::with_capacity(winners.len()),
            options: Vec::with_capacity(winners.len()),
            reduced: Vec::with_capacity(winners.len()),
            costs: Vec::with_capacity(winners.len()),
            gains: Vec::with_capacity(winners.len()),
            starts: vec![0],
            rows: Vec::new(),
            values: Vec::new(),
            repair,
            minimum,
            violations,
            evaluated,
        };
        for winner in winners {
            let id = ((winner.session as u64) << 4) | winner.option as u64;
            let signature = self.option_signatures[winner.option] as usize;
            let duration =
                self.features[(winner.session * self.signatures + signature) * FEATURES + 4];
            sweep.ids.push(id);
            sweep.sessions.push(winner.session as i32);
            sweep.options.push(winner.option as u16);
            sweep.reduced.push(winner.reduced);
            sweep.costs.push(duration / self.horizon);
            sweep.gains.push(self.gains[winner.session]);
            let start = self.option_starts[winner.option] as usize;
            let end = self.option_starts[winner.option + 1] as usize;
            for entry in start..end {
                let value = self.value(winner.session, winner.option, entry);
                if value != 0.0 {
                    sweep.rows.push(self.resource_rows[entry]);
                    sweep.values.push(value);
                }
            }
            sweep.starts.push(sweep.rows.len() as i32);
        }
        if !sweep.ids.is_empty() {
            self.pending = Some((epoch, sweep.ids.clone()));
        }
        Ok(sweep)
    }
}

#[pymethods]
impl PricingOracle {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        sessions: usize,
        signatures: usize,
        options: usize,
        resources: usize,
        horizon: f64,
        gains: PyReadonlyArray1<'_, f64>,
        features: PyReadonlyArray1<'_, f64>,
        feasible: PyReadonlyArray1<'_, u16>,
        option_signatures: PyReadonlyArray1<'_, u16>,
        option_starts: PyReadonlyArray1<'_, i32>,
        resource_rows: PyReadonlyArray1<'_, i32>,
        coefficients: PyReadonlyArray1<'_, f64>,
        session_ranks: PyReadonlyArray1<'_, u32>,
    ) -> PyResult<Self> {
        let gains = gains.as_slice()?.to_vec();
        let features = features.as_slice()?.to_vec();
        let feasible = feasible.as_slice()?.to_vec();
        let option_signatures = option_signatures.as_slice()?.to_vec();
        let option_starts = option_starts.as_slice()?.to_vec();
        let resource_rows = resource_rows.as_slice()?.to_vec();
        let coefficients = coefficients.as_slice()?.to_vec();
        let session_ranks = session_ranks.as_slice()?.to_vec();
        let feature_count = sessions
            .checked_mul(signatures)
            .and_then(|value| value.checked_mul(FEATURES));
        let mut seen_ranks = vec![false; sessions];
        let ranks_valid = session_ranks.iter().all(|rank| {
            let rank = *rank as usize;
            if rank >= sessions || seen_ranks[rank] {
                false
            } else {
                seen_ranks[rank] = true;
                true
            }
        });
        let valid_mask = if options >= 16 {
            u16::MAX
        } else {
            (1u16 << options) - 1
        };
        if options > 16
            || options == 0
            || signatures == 0
            || feature_count.is_none()
            || sessions > i32::MAX as usize
            || resources > i32::MAX as usize
            || resource_rows.len() > i32::MAX as usize
            || !horizon.is_finite()
            || horizon <= 0.0
            || gains.len() != sessions
            || feasible.len() != sessions
            || session_ranks.len() != sessions
            || feature_count != Some(features.len())
            || option_signatures.len() != options
            || option_starts.len() != options + 1
            || resource_rows.len().checked_mul(FEATURES) != Some(coefficients.len())
            || option_starts.first().copied() != Some(0)
            || option_starts
                .windows(2)
                .any(|window| window[0] < 0 || window[0] > window[1])
            || option_starts.last().copied() != Some(resource_rows.len() as i32)
            || feasible.iter().any(|mask| mask & !valid_mask != 0)
            || !ranks_valid
            || option_signatures
                .iter()
                .any(|value| *value as usize >= signatures)
            || resource_rows
                .iter()
                .any(|value| *value < 0 || *value as usize >= resources)
            || !gains.iter().all(|value| value.is_finite())
            || !features.iter().all(|value| value.is_finite())
            || !coefficients
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        {
            return Err(PyValueError::new_err("invalid pricing SoA"));
        }
        Ok(Self {
            sessions,
            signatures,
            options,
            resources,
            horizon,
            gains,
            features,
            feasible,
            option_signatures,
            option_starts,
            resource_rows,
            coefficients,
            session_ranks,
            generated: vec![0; sessions],
            loaded: vec![true; sessions],
            rank_used: vec![true; sessions],
            loaded_count: sessions,
            epoch: 0,
            pending: None,
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn allocate(
        sessions: usize,
        signatures: usize,
        options: usize,
        resources: usize,
        horizon: f64,
        option_signatures: PyReadonlyArray1<'_, u16>,
        option_starts: PyReadonlyArray1<'_, i32>,
        resource_rows: PyReadonlyArray1<'_, i32>,
        coefficients: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let option_signatures = option_signatures.as_slice()?.to_vec();
        let option_starts = option_starts.as_slice()?.to_vec();
        let resource_rows = resource_rows.as_slice()?.to_vec();
        let coefficients = coefficients.as_slice()?.to_vec();
        let feature_count = sessions
            .checked_mul(signatures)
            .and_then(|value| value.checked_mul(FEATURES));
        if options == 0
            || options > 16
            || signatures == 0
            || feature_count.is_none()
            || sessions > i32::MAX as usize
            || resources > i32::MAX as usize
            || resource_rows.len() > i32::MAX as usize
            || !horizon.is_finite()
            || horizon <= 0.0
            || option_signatures.len() != options
            || option_starts.len() != options + 1
            || resource_rows.len().checked_mul(FEATURES) != Some(coefficients.len())
            || option_starts.first().copied() != Some(0)
            || option_starts
                .windows(2)
                .any(|window| window[0] < 0 || window[0] > window[1])
            || option_starts.last().copied() != Some(resource_rows.len() as i32)
            || option_signatures
                .iter()
                .any(|value| *value as usize >= signatures)
            || resource_rows
                .iter()
                .any(|value| *value < 0 || *value as usize >= resources)
            || !coefficients
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        {
            return Err(PyValueError::new_err("invalid pricing SoA"));
        }
        Ok(Self {
            sessions,
            signatures,
            options,
            resources,
            horizon,
            gains: vec![0.0; sessions],
            features: vec![0.0; feature_count.unwrap()],
            feasible: vec![0; sessions],
            option_signatures,
            option_starts,
            resource_rows,
            coefficients,
            session_ranks: vec![0; sessions],
            generated: vec![0; sessions],
            loaded: vec![false; sessions],
            rank_used: vec![false; sessions],
            loaded_count: 0,
            epoch: 0,
            pending: None,
        })
    }

    fn load(
        &mut self,
        start: usize,
        gains: PyReadonlyArray1<'_, f64>,
        features: PyReadonlyArray1<'_, f64>,
        feasible: PyReadonlyArray1<'_, u16>,
        session_ranks: PyReadonlyArray1<'_, u32>,
    ) -> PyResult<()> {
        let gains = gains.as_slice()?;
        let features = features.as_slice()?;
        let feasible = feasible.as_slice()?;
        let session_ranks = session_ranks.as_slice()?;
        let count = gains.len();
        let end = start.checked_add(count);
        let valid_mask = if self.options == 16 {
            u16::MAX
        } else {
            (1u16 << self.options) - 1
        };
        let mut ranks: Vec<_> = session_ranks.iter().map(|rank| *rank as usize).collect();
        ranks.sort_unstable();
        if end.is_none_or(|end| end > self.sessions)
            || features.len()
                != count
                    .checked_mul(self.signatures)
                    .and_then(|value| value.checked_mul(FEATURES))
                    .unwrap_or(usize::MAX)
            || feasible.len() != count
            || session_ranks.len() != count
            || !gains.iter().all(|value| value.is_finite())
            || !features.iter().all(|value| value.is_finite())
            || feasible.iter().any(|mask| mask & !valid_mask != 0)
            || (start..end.unwrap()).any(|session| self.loaded[session])
            || ranks
                .iter()
                .any(|rank| *rank >= self.sessions || self.rank_used[*rank])
            || ranks.windows(2).any(|window| window[0] == window[1])
        {
            return Err(PyValueError::new_err("invalid pricing SoA chunk"));
        }
        let end = end.unwrap();
        self.gains[start..end].copy_from_slice(gains);
        self.feasible[start..end].copy_from_slice(feasible);
        self.session_ranks[start..end].copy_from_slice(session_ranks);
        let feature_start = start * self.signatures * FEATURES;
        self.features[feature_start..feature_start + features.len()].copy_from_slice(features);
        for session in start..end {
            self.loaded[session] = true;
        }
        for rank in ranks {
            self.rank_used[rank] = true;
        }
        self.loaded_count += count;
        Ok(())
    }

    fn price<'py>(
        &mut self,
        py: Python<'py>,
        phase: u8,
        eta: f64,
        resource_duals: PyReadonlyArray1<'_, f64>,
        session_duals: PyReadonlyArray1<'_, f64>,
        batch: usize,
        tolerance: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let resource_duals = resource_duals.as_slice()?.to_vec();
        let session_duals = session_duals.as_slice()?.to_vec();
        let sweep = py.detach(|| {
            self.price_inner(
                phase,
                eta,
                &resource_duals,
                &session_duals,
                batch,
                tolerance,
            )
        })?;
        let result = PyDict::new(py);
        result.set_item("epoch", sweep.epoch)?;
        result.set_item("effective_eta", sweep.eta)?;
        result.set_item("candidate_ids", PyArray1::from_vec(py, sweep.ids))?;
        result.set_item("session_indices", PyArray1::from_vec(py, sweep.sessions))?;
        result.set_item("option_indices", PyArray1::from_vec(py, sweep.options))?;
        result.set_item("reduced_costs", PyArray1::from_vec(py, sweep.reduced))?;
        result.set_item("phase2_costs", PyArray1::from_vec(py, sweep.costs))?;
        result.set_item("gains", PyArray1::from_vec(py, sweep.gains))?;
        result.set_item("resource_starts", PyArray1::from_vec(py, sweep.starts))?;
        result.set_item("resource_rows", PyArray1::from_vec(py, sweep.rows))?;
        result.set_item("resource_values", PyArray1::from_vec(py, sweep.values))?;
        result.set_item("repair_sum", sweep.repair)?;
        result.set_item("minimum_reduced_cost", sweep.minimum)?;
        result.set_item("violating_sessions", sweep.violations)?;
        result.set_item("evaluated_choices", sweep.evaluated)?;
        Ok(result)
    }

    fn commit(&mut self, epoch: u64, candidate_ids: PyReadonlyArray1<'_, u64>) -> PyResult<()> {
        let ids = candidate_ids.as_slice()?;
        match self.pending.take() {
            Some((pending_epoch, pending)) if pending_epoch == epoch && pending == ids => {
                for id in ids {
                    let session = (id >> 4) as usize;
                    let option = (id & 15) as usize;
                    self.generated[session] |= 1u16 << option;
                }
                Ok(())
            }
            pending => {
                self.pending = pending;
                Err(PyValueError::new_err(
                    "pricing commit does not match pending batch",
                ))
            }
        }
    }

    fn discard(&mut self, epoch: u64) -> PyResult<()> {
        match self.pending.take() {
            Some((pending_epoch, _)) if pending_epoch == epoch => Ok(()),
            pending => {
                self.pending = pending;
                Err(PyValueError::new_err(
                    "pricing discard does not match pending batch",
                ))
            }
        }
    }
}

#[pymodule]
fn _queue_haul_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PricingOracle>()
}

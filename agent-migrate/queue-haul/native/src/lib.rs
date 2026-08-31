use std::cmp::Ordering;
use std::collections::BinaryHeap;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

const FEATURES: usize = 7;

fn greedy_order(costs: &[f64], gains: &[f64], candidates: &[usize], recovery: u8) -> Vec<usize> {
    let mut order = candidates.to_vec();
    order.sort_unstable_by(|a, b| {
        (recovery == 2)
            .then(|| gains[*b].total_cmp(&gains[*a]))
            .unwrap_or(Ordering::Equal)
            .then((gains[*b] / costs[*b].max(1e-12)).total_cmp(&(gains[*a] / costs[*a].max(1e-12))))
            .then_with(|| {
                (recovery == 1)
                    .then(|| costs[*a].total_cmp(&costs[*b]))
                    .unwrap_or(Ordering::Equal)
            })
            .then(a.cmp(b))
    });
    order
}

fn greedy_scan<F>(
    target: f64,
    session_count: usize,
    resource_count: usize,
    sessions: &[i32],
    gains: &[f64],
    order: &[usize],
    state: &[usize],
    mut column: F,
) -> (Vec<usize>, f64)
where
    F: FnMut(usize, &mut dyn FnMut(usize, f64)),
{
    let mut usage = vec![0.0; resource_count];
    let mut taken = vec![false; session_count];
    let mut selected = state.to_vec();
    let mut gain = 0.0;
    for candidate in state {
        taken[sessions[*candidate] as usize] = true;
        gain += gains[*candidate];
        column(*candidate, &mut |row, value| usage[row] += value);
    }
    for candidate in order {
        if gain >= target - 1e-8 {
            break;
        }
        let session = sessions[*candidate] as usize;
        let mut fits = !taken[session];
        column(*candidate, &mut |row, value| {
            fits &= usage[row] + value <= 1.0 + 1e-8
        });
        if !fits {
            continue;
        }
        column(*candidate, &mut |row, value| usage[row] += value);
        taken[session] = true;
        gain += gains[*candidate];
        selected.push(*candidate);
    }
    (selected, gain)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn greedy_compact<'py>(
    py: Python<'py>,
    session_count: usize,
    resource_count: usize,
    target: f64,
    sessions: PyReadonlyArray1<'_, i32>,
    options: PyReadonlyArray1<'_, i32>,
    session_gains: PyReadonlyArray1<'_, f64>,
    features: PyReadonlyArray1<'_, f64>,
    option_starts: PyReadonlyArray1<'_, i32>,
    resource_rows: PyReadonlyArray1<'_, i32>,
    coefficients: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let sessions = sessions.as_slice()?;
    let options = options.as_slice()?;
    let session_gains = session_gains.as_slice()?;
    let features = features.as_slice()?;
    let option_starts = option_starts.as_slice()?;
    let resource_rows = resource_rows.as_slice()?;
    let coefficients = coefficients.as_slice()?;
    let candidates = sessions.len();
    let option_count = option_starts.len().saturating_sub(1);
    if !target.is_finite()
        || target < 0.0
        || options.len() != candidates
        || features.len() != candidates.checked_mul(FEATURES).unwrap_or(usize::MAX)
        || session_gains.len() != session_count
        || resource_rows.len().checked_mul(FEATURES) != Some(coefficients.len())
        || option_starts.first().copied() != Some(0)
        || option_starts
            .windows(2)
            .any(|window| window[0] < 0 || window[0] > window[1])
        || option_starts.last().copied() != Some(resource_rows.len() as i32)
        || sessions
            .iter()
            .any(|value| *value < 0 || *value as usize >= session_count)
        || options
            .iter()
            .any(|value| *value < 0 || *value as usize >= option_count)
        || resource_rows
            .iter()
            .any(|value| *value < 0 || *value as usize >= resource_count)
        || !session_gains.iter().all(|value| value.is_finite())
        || !features.iter().all(|value| value.is_finite())
        || !coefficients
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
    {
        return Err(PyValueError::new_err("invalid compact greedy problem"));
    }
    let value = |candidate: usize, entry: usize| {
        (0..FEATURES)
            .map(|feature| {
                features[candidate * FEATURES + feature] * coefficients[entry * FEATURES + feature]
            })
            .sum::<f64>()
    };
    for option in 0..option_count {
        let rows =
            &resource_rows[option_starts[option] as usize..option_starts[option + 1] as usize];
        if rows
            .iter()
            .enumerate()
            .any(|(i, row)| rows[i + 1..].contains(row))
        {
            return Err(PyValueError::new_err(
                "compact greedy option repeats a resource",
            ));
        }
    }
    for candidate in 0..candidates {
        let option = options[candidate] as usize;
        for entry in option_starts[option] as usize..option_starts[option + 1] as usize {
            let resource = value(candidate, entry);
            if !resource.is_finite() || resource < 0.0 {
                return Err(PyValueError::new_err(
                    "invalid compact greedy resource value",
                ));
            }
        }
    }
    let visit = |candidate: usize, emit: &mut dyn FnMut(usize, f64)| {
        let option = options[candidate] as usize;
        for entry in option_starts[option] as usize..option_starts[option + 1] as usize {
            emit(resource_rows[entry] as usize, value(candidate, entry));
        }
    };
    let mut cheapest: Vec<Option<(f64, usize)>> = vec![None; session_count];
    for candidate in 0..candidates {
        let mut demand = 0.0;
        visit(candidate, &mut |_row, value| demand += value);
        let session = sessions[candidate] as usize;
        if cheapest[session].is_none_or(|current| (demand, candidate) < current) {
            cheapest[session] = Some((demand, candidate));
        }
    }
    let mut prices = vec![0.0_f64; resource_count];
    for (_, candidate) in cheapest.into_iter().flatten() {
        visit(candidate, &mut |row, value| prices[row] += value);
    }
    prices.iter_mut().for_each(|price| *price = price.max(1.0));
    let gains: Vec<_> = sessions
        .iter()
        .map(|session| session_gains[*session as usize])
        .collect();
    let costs: Vec<_> = (0..candidates)
        .map(|candidate| {
            let mut cost = 0.0;
            visit(candidate, &mut |row, value| cost += prices[row] * value);
            cost
        })
        .collect();
    let indices: Vec<_> = (0..candidates).collect();
    let original = greedy_order(&costs, &gains, &indices, 0);
    let (mut selected, mut gain) = greedy_scan(
        target,
        session_count,
        resource_count,
        sessions,
        &gains,
        &original,
        &[],
        visit,
    );
    if gain < target - 1e-8 {
        let retry = greedy_order(&costs, &gains, &indices, 1);
        let (alternative, alternative_gain) = greedy_scan(
            target,
            session_count,
            resource_count,
            sessions,
            &gains,
            &retry,
            &[],
            visit,
        );
        if alternative_gain >= target - 1e-8 {
            selected = alternative;
            gain = alternative_gain;
        }
    }
    if gain < target - 1e-8 {
        let retry = greedy_order(&costs, &gains, &indices, 2);
        let (alternative, alternative_gain) = greedy_scan(
            target,
            session_count,
            resource_count,
            sessions,
            &gains,
            &retry,
            &[],
            visit,
        );
        if alternative_gain >= target - 1e-8 {
            selected = alternative;
        }
    }
    Ok(PyArray1::from_vec(
        py,
        selected.into_iter().map(|value| value as i64).collect(),
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn greedy_csc<'py>(
    py: Python<'py>,
    session_count: usize,
    resource_count: usize,
    target: f64,
    sessions: PyReadonlyArray1<'_, i32>,
    gains: PyReadonlyArray1<'_, f64>,
    starts: PyReadonlyArray1<'_, i32>,
    rows: PyReadonlyArray1<'_, i32>,
    values: PyReadonlyArray1<'_, f64>,
    eligible: PyReadonlyArray1<'_, i64>,
    state: PyReadonlyArray1<'_, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let sessions = sessions.as_slice()?;
    let gains = gains.as_slice()?;
    let starts = starts.as_slice()?;
    let rows = rows.as_slice()?;
    let values = values.as_slice()?;
    let eligible = eligible.as_slice()?;
    let state = state.as_slice()?;
    let candidates = sessions.len();
    if !target.is_finite()
        || target < 0.0
        || gains.len() != candidates
        || starts.len() != candidates + 1
        || starts.first().copied() != Some(0)
        || starts
            .windows(2)
            .any(|window| window[0] < 0 || window[0] > window[1])
        || starts.last().copied() != Some(rows.len() as i32)
        || rows.len() != values.len()
        || sessions
            .iter()
            .any(|value| *value < 0 || *value as usize >= session_count)
        || rows
            .iter()
            .any(|value| *value < 0 || *value as usize >= resource_count)
        || eligible
            .iter()
            .chain(state)
            .any(|value| *value < 0 || *value as usize >= candidates)
        || !gains.iter().all(|value| value.is_finite())
        || !values
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
    {
        return Err(PyValueError::new_err("invalid sparse greedy problem"));
    }
    let eligible: Vec<_> = eligible.iter().map(|value| *value as usize).collect();
    let state: Vec<_> = state.iter().map(|value| *value as usize).collect();
    let visit = |candidate: usize, emit: &mut dyn FnMut(usize, f64)| {
        for entry in starts[candidate] as usize..starts[candidate + 1] as usize {
            emit(rows[entry] as usize, values[entry]);
        }
    };
    let mut cheapest: Vec<Option<(f64, usize)>> = vec![None; session_count];
    for candidate in &eligible {
        let mut demand = 0.0;
        visit(*candidate, &mut |_row, value| demand += value);
        let session = sessions[*candidate] as usize;
        if cheapest[session].is_none_or(|current| (demand, *candidate) < current) {
            cheapest[session] = Some((demand, *candidate));
        }
    }
    let mut prices = vec![0.0_f64; resource_count];
    for (_, candidate) in cheapest.into_iter().flatten() {
        visit(candidate, &mut |row, value| prices[row] += value);
    }
    prices.iter_mut().for_each(|price| *price = price.max(1.0));
    let mut costs = vec![0.0; candidates];
    for candidate in &eligible {
        visit(*candidate, &mut |row, value| {
            costs[*candidate] += prices[row] * value
        });
    }
    let original = greedy_order(&costs, gains, &eligible, 0);
    let (mut selected, mut gain) = greedy_scan(
        target,
        session_count,
        resource_count,
        sessions,
        gains,
        &original,
        &state,
        visit,
    );
    if gain < target - 1e-8 {
        let retry = greedy_order(&costs, gains, &eligible, 1);
        let (alternative, alternative_gain) = greedy_scan(
            target,
            session_count,
            resource_count,
            sessions,
            gains,
            &retry,
            &state,
            visit,
        );
        if alternative_gain >= target - 1e-8 {
            selected = alternative;
            gain = alternative_gain;
        }
    }
    if gain < target - 1e-8 {
        let retry = greedy_order(&costs, gains, &eligible, 2);
        let (alternative, alternative_gain) = greedy_scan(
            target,
            session_count,
            resource_count,
            sessions,
            gains,
            &retry,
            &state,
            visit,
        );
        if alternative_gain >= target - 1e-8 {
            selected = alternative;
        }
    }
    Ok(PyArray1::from_vec(
        py,
        selected.into_iter().map(|value| value as i64).collect(),
    ))
}

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
    features: Vec<f64>,
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
            features: Vec::with_capacity(winners.len() * FEATURES),
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
            let feature = (winner.session * self.signatures + signature) * FEATURES;
            sweep
                .features
                .extend_from_slice(&self.features[feature..feature + FEATURES]);
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
        result.set_item("candidate_features", PyArray1::from_vec(py, sweep.features))?;
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
    module.add_class::<PricingOracle>()?;
    module.add_function(wrap_pyfunction!(greedy_compact, module)?)?;
    module.add_function(wrap_pyfunction!(greedy_csc, module)?)?;
    Ok(())
}

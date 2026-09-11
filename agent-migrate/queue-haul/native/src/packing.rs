use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

struct Packing<'a> {
    starts: &'a [i32],
    rows: &'a [i32],
    values: &'a [f64],
    gains: &'a [f64],
    secondary: &'a [f64],
}

impl Packing<'_> {
    fn column(&self, j: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        (self.starts[j] as usize..self.starts[j + 1] as usize)
            .map(|k| (self.rows[k] as usize, self.values[k]))
    }

    fn usage(&self, x: &[f64], resources: usize) -> Vec<f64> {
        let mut usage = vec![0.; resources];
        for (j, mass) in x.iter().enumerate() {
            for (i, value) in self.column(j) {
                usage[i] += value * mass;
            }
        }
        usage
    }

    fn gain(&self, x: &[f64]) -> f64 {
        x.iter().zip(self.gains).map(|(x, c)| x * c).sum()
    }

    fn offer(&self, j: usize, remaining: &[f64]) -> PyResult<Option<(f64, f64)>> {
        let mut take = f64::INFINITY;
        for (i, value) in self.column(j) {
            if remaining[i] <= 1e-10 {
                return Ok(None);
            }
            take = take.min(remaining[i] / value);
        }
        let score = self.gains[j] * take;
        if !score.is_finite() {
            return Err(PyValueError::new_err("nonfinite packing greedy score"));
        }
        Ok(Some((score, take)))
    }

    fn fill(&self, x: &mut [f64], remaining: &mut [f64]) -> PyResult<()> {
        // Nonnegative finite scores have the same bit and numerical ordering.
        let mut heap = BinaryHeap::new();
        for j in 0..x.len() {
            if let Some((score, _)) = self.offer(j, remaining)? {
                heap.push((score.to_bits(), Reverse(j)));
            }
        }
        let mut commits = 0;
        while let Some((_, Reverse(j))) = heap.pop() {
            let Some((maximum, take)) = self.offer(j, remaining)? else {
                continue;
            };
            if heap
                .peek()
                .is_some_and(|(score, _)| f64::from_bits(*score) > maximum)
            {
                heap.push((maximum.to_bits(), Reverse(j)));
                continue;
            }
            let mut tied = vec![(j, maximum, take)];
            while heap
                .peek()
                .is_some_and(|(score, _)| maximum - f64::from_bits(*score) <= 1e-12 * maximum)
            {
                let (_, Reverse(k)) = heap.pop().unwrap();
                if let Some((score, take)) = self.offer(k, remaining)? {
                    if maximum - score <= 1e-12 * maximum {
                        tied.push((k, score, take));
                    } else {
                        heap.push((score.to_bits(), Reverse(k)));
                    }
                }
            }
            let secondary = tied
                .iter()
                .map(|(j, _, _)| self.secondary[*j])
                .fold(f64::INFINITY, f64::min);
            let &(j, _, take) = tied
                .iter()
                .filter(|(j, _, _)| (self.secondary[*j] - secondary).abs() <= 1e-12 * secondary)
                .min_by_key(|(j, _, _)| *j)
                .unwrap();
            x[j] += take;
            for (i, value) in self.column(j) {
                remaining[i] = (remaining[i] - take * value).max(0.);
            }
            for (k, score, _) in tied {
                if k != j {
                    heap.push((score.to_bits(), Reverse(k)));
                }
            }
            commits += 1;
            if commits > remaining.len() {
                return Err(PyValueError::new_err(
                    "packing greedy failed to exhaust a resource",
                ));
            }
        }
        Ok(())
    }

    fn compress(&self, x: &mut [f64], resources: usize) -> PyResult<()> {
        let objective = self.gain(x);
        let support: Vec<usize> = (0..x.len()).filter(|j| x[*j] > 0.).collect();
        let mut usage = self.usage(x, resources);
        let mut local = vec![usize::MAX; resources];
        let mut begin = 0;
        while begin < support.len() {
            let mut end = begin;
            let mut touched = Vec::new();
            while end < support.len() {
                let new = self
                    .column(support[end])
                    .filter(|(i, _)| local[*i] == usize::MAX)
                    .count();
                // ponytail: bounded local bases; a wider singleton stays unchanged.
                if end > begin && touched.len() + new > 128 {
                    break;
                }
                for (i, _) in self.column(support[end]) {
                    if local[i] == usize::MAX {
                        local[i] = 0;
                        touched.push(i);
                    }
                }
                end += 1;
                if touched.len() > 128 {
                    break;
                }
            }
            if end - begin > 1 {
                touched.sort_unstable();
                for (i, &global) in touched.iter().enumerate() {
                    local[global] = i;
                }
                let ids = &support[begin..end];
                let mut consumed = vec![0.; touched.len()];
                for &j in ids {
                    for (i, value) in self.column(j) {
                        consumed[local[i]] += value * x[j];
                    }
                }
                let capacity: Vec<f64> = touched
                    .iter()
                    .enumerate()
                    .map(|(i, &global)| (1. - usage[global]) + consumed[i])
                    .collect();
                if capacity.iter().any(|v| !v.is_finite() || *v <= 0.) {
                    return Err(PyValueError::new_err(
                        "invalid packing compression residual",
                    ));
                }
                let rows = touched.len();
                let mut columns = Vec::with_capacity(ids.len());
                let mut gains = Vec::with_capacity(ids.len());
                let mut upper = Vec::with_capacity(ids.len());
                let mut weights = Vec::with_capacity(ids.len());
                for &j in ids {
                    let mut column: Vec<(usize, f64)> = self
                        .column(j)
                        .map(|(i, value)| (local[i], value / capacity[local[i]]))
                        .collect();
                    let maximum = column.iter().map(|(_, v)| *v).fold(0., f64::max);
                    let limit = 1. / maximum;
                    for (_, v) in &mut column {
                        *v /= maximum;
                    }
                    columns.push(column);
                    upper.push(limit);
                    gains.push(self.gains[j] * limit);
                    weights.push(x[j] / limit);
                }
                reduce_support(&columns, &gains, &mut weights, rows)?;
                let mut after = vec![0.; touched.len()];
                for (k, &j) in ids.iter().enumerate() {
                    x[j] = weights[k] * upper[k];
                    for (i, value) in self.column(j) {
                        after[local[i]] += value * x[j];
                    }
                }
                for (i, &global) in touched.iter().enumerate() {
                    usage[global] += after[i] - consumed[i];
                }
            }
            for i in touched {
                local[i] = usize::MAX;
            }
            begin = end;
        }
        if x.iter().any(|v| !v.is_finite() || *v < 0.)
            || (self.gain(x) - objective).abs() > 1e-9 * objective
            || self
                .usage(x, resources)
                .iter()
                .any(|v| !v.is_finite() || *v > 1. + 1e-9)
        {
            return Err(PyValueError::new_err(
                "packing compression changed objective or feasibility",
            ));
        }
        Ok(())
    }
}

fn reduce_support(
    columns: &[Vec<(usize, f64)>],
    gains: &[f64],
    weights: &mut [f64],
    rows: usize,
) -> PyResult<()> {
    let n = weights.len();
    let p = rows + 1;
    let scale = gains.iter().copied().fold(0., f64::max);
    let gains: Vec<f64> = gains.iter().map(|v| v / scale).collect();
    if gains
        .iter()
        .chain(weights.iter())
        .any(|v| !v.is_finite() || *v <= 0.)
        || columns
            .iter()
            .flatten()
            .any(|(_, v)| !v.is_finite() || *v <= 0. || *v > 1.)
    {
        return Err(PyValueError::new_err(
            "packing compression normalization exceeds finite precision",
        ));
    }
    let star = gains.iter().position(|v| *v == 1.).unwrap();
    let mut z = weights.to_vec();
    z.extend(vec![1.; rows]);
    for (j, column) in columns.iter().enumerate() {
        for &(i, value) in column {
            z[n + i] -= value * weights[j];
        }
    }
    for value in &mut z[n..] {
        if !value.is_finite() || *value < -1e-9 {
            return Err(PyValueError::new_err(
                "infeasible packing compression input",
            ));
        }
        *value = value.max(0.);
    }
    let mut basis: Vec<usize> = (n..n + rows).chain(std::iter::once(star)).collect();
    let mut inverse = vec![0.; p * p];
    for i in 0..p {
        inverse[i * p + i] = 1.;
    }
    for &(i, value) in &columns[star] {
        inverse[rows * p + i] = -value;
    }
    let mut direction = vec![0.; p];
    let mut pivot_row = vec![0.; p];
    for j in 0..n {
        if j == star {
            continue;
        }
        direction.fill(0.);
        for &(r, value) in &columns[j] {
            for i in 0..p {
                direction[i] += inverse[r * p + i] * value;
            }
        }
        for i in 0..p {
            direction[i] += inverse[rows * p + i] * gains[j];
        }
        let mass = z[j];
        let mut alpha = mass;
        let mut limiting: Option<usize> = None;
        for i in 0..p {
            if !direction[i].is_finite() {
                return Err(PyValueError::new_err(
                    "nonfinite packing compression direction",
                ));
            }
            if direction[i] < -1e-12 {
                let ratio = z[basis[i]].max(0.) / -direction[i];
                if ratio < alpha
                    || (ratio == alpha
                        && limiting.is_some_and(|k| direction[i].abs() > direction[k].abs()))
                {
                    alpha = ratio;
                    limiting = Some(i);
                }
            }
        }
        for i in 0..p {
            z[basis[i]] += alpha * direction[i];
            if !z[basis[i]].is_finite() || z[basis[i]] < -1e-9 {
                return Err(PyValueError::new_err("unstable packing compression pivot"));
            }
        }
        z[j] -= alpha;
        if alpha == mass {
            z[j] = 0.;
            continue;
        }
        let leaving =
            limiting.ok_or_else(|| PyValueError::new_err("missing packing compression pivot"))?;
        let pivot = direction[leaving];
        z[basis[leaving]] = 0.;
        for k in 0..p {
            pivot_row[k] = inverse[k * p + leaving] / pivot;
        }
        for k in 0..p {
            for i in 0..p {
                inverse[k * p + i] -= direction[i] * pivot_row[k];
                if !inverse[k * p + i].is_finite() {
                    return Err(PyValueError::new_err(
                        "nonfinite packing compression inverse",
                    ));
                }
            }
            inverse[k * p + leaving] = pivot_row[k];
        }
        basis[leaving] = j;
    }
    for (output, value) in weights.iter_mut().zip(z) {
        *output = value.max(0.);
    }
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn packing_coordinate<'py>(
    py: Python<'py>,
    resource_count: usize,
    candidate_count: usize,
    starts: PyReadonlyArray1<'_, i32>,
    rows: PyReadonlyArray1<'_, i32>,
    values: PyReadonlyArray1<'_, f64>,
    gains: PyReadonlyArray1<'_, f64>,
    incumbent: PyReadonlyArray1<'_, f64>,
    secondary: PyReadonlyArray1<'_, f64>,
    max_iterations: usize,
    relative_tolerance: f64,
    absolute_tolerance: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, usize)> {
    let (starts, rows, values, gains, incumbent, secondary) = (
        starts.as_slice()?,
        rows.as_slice()?,
        values.as_slice()?,
        gains.as_slice()?,
        incumbent.as_slice()?,
        secondary.as_slice()?,
    );
    if starts.len() != candidate_count + 1
        || starts.first() != Some(&0)
        || starts.last().copied().map(i64::from) != Some(rows.len() as i64)
        || starts.windows(2).any(|w| w[0] < 0 || w[0] >= w[1])
        || values.len() != rows.len()
        || gains.len() != candidate_count
        || incumbent.len() != candidate_count
        || secondary.len() != candidate_count
        || rows.iter().any(|i| *i < 0 || *i as usize >= resource_count)
        || values.iter().any(|v| !v.is_finite() || *v <= 0. || *v > 1.)
        || gains.iter().any(|v| !v.is_finite() || *v <= 0.)
        || incumbent
            .iter()
            .chain(secondary)
            .any(|v| !v.is_finite() || *v < 0.)
        || !relative_tolerance.is_finite()
        || !(0.0..1.0).contains(&relative_tolerance)
        || !absolute_tolerance.is_finite()
        || absolute_tolerance < 0.
    {
        return Err(PyValueError::new_err(
            "invalid normalized CSC packing problem",
        ));
    }
    if (0..candidate_count).any(|j| {
        rows[starts[j] as usize..starts[j + 1] as usize]
            .windows(2)
            .any(|w| w[0] >= w[1])
    }) {
        return Err(PyValueError::new_err(
            "packing column rows must be sorted and unique",
        ));
    }
    let problem = Packing {
        starts,
        rows,
        values,
        gains,
        secondary,
    };
    let mut best = incumbent.to_vec();
    let mut usage = problem.usage(&best, resource_count);
    if usage.iter().any(|v| !v.is_finite() || *v > 1. + 1e-8) {
        return Err(PyValueError::new_err("infeasible packing incumbent"));
    }
    let scale = usage.iter().copied().fold(1., f64::max);
    best.iter_mut().for_each(|v| *v /= scale);
    let mut seed = vec![0.; candidate_count];
    problem.fill(&mut seed, &mut vec![1.; resource_count])?;
    if problem.gain(&seed) > problem.gain(&best) {
        best = seed;
    }
    let price = (0..candidate_count)
        .map(|j| gains[j] / problem.column(j).map(|(_, v)| v).sum::<f64>())
        .fold(0., f64::max)
        * (1. + 1e-12);
    let mut witness = vec![price; resource_count];
    let mut bound: f64 = witness.iter().sum();
    let mut cover = vec![0.; resource_count];
    for j in 0..candidate_count {
        let (mut covered, mut largest, mut resource) = (0., 0., 0);
        for (i, value) in problem.column(j) {
            covered += value * cover[i];
            if value > largest {
                largest = value;
                resource = i;
            }
        }
        if !covered.is_finite() {
            return Err(PyValueError::new_err("nonfinite packing covering price"));
        }
        if covered < gains[j] {
            cover[resource] += (gains[j] - covered) / largest * (1. + 1e-12);
            if !cover[resource].is_finite() {
                return Err(PyValueError::new_err(
                    "nonfinite packing covering increment",
                ));
            }
        }
    }
    let cover_bound: f64 = cover.iter().sum();
    if !cover_bound.is_finite() {
        return Err(PyValueError::new_err("nonfinite packing covering bound"));
    }
    if cover_bound < bound {
        bound = cover_bound;
        witness = cover;
    }
    let mut objective = problem.gain(&best);
    if !bound.is_finite() || !objective.is_finite() {
        return Err(PyValueError::new_err(
            "packing certificate exceeds finite precision",
        ));
    }
    let rho = 0.5 * gains.iter().copied().fold(0., f64::max);
    let price_floor = (2e-16 * rho / resource_count.max(1) as f64).max(f64::from_bits(1));
    // Fixed diagonal penalties balance resource rows with small coefficients.
    let mut weights = vec![1. / candidate_count.max(1) as f64; resource_count];
    for (&i, &value) in rows.iter().zip(values) {
        weights[i as usize] = weights[i as usize].max(value);
    }
    for weight in &mut weights {
        *weight = 1. / *weight;
    }
    let norms: Vec<f64> = (0..candidate_count)
        .map(|j| problem.column(j).map(|(i, v)| weights[i] * v * v).sum())
        .collect();
    if candidate_count > 0 && (rho <= 0. || norms.iter().any(|q| !q.is_finite() || *q <= 0.)) {
        return Err(PyValueError::new_err(
            "packing coordinate scale exceeds finite precision",
        ));
    }
    let mut x = best.clone();
    usage = problem.usage(&x, resource_count);
    let mut dual = vec![0.; resource_count];
    let mut residual = vec![0.; resource_count];
    let mut active = vec![true; candidate_count];
    let mut iterations = 0;
    for iteration in 0..max_iterations {
        if bound - objective <= absolute_tolerance.max(relative_tolerance * bound) {
            break;
        }
        for i in 0..resource_count {
            let slack = 1. - usage[i] - dual[i] / (rho * weights[i]);
            if !slack.is_finite() {
                return Err(PyValueError::new_err("nonfinite packing slack"));
            }
            residual[i] = usage[i] + slack.max(0.) - 1.;
        }
        for sweep in 0..2 {
            for z in 0..candidate_count {
                let j = if (iteration + sweep) % 2 == 0 {
                    z
                } else {
                    candidate_count - z - 1
                };
                if !active[j] && x[j] == 0. {
                    continue;
                }
                let priced: f64 = problem
                    .column(j)
                    .map(|(i, v)| v * (dual[i] + rho * weights[i] * residual[i]))
                    .sum();
                // x_j <- max(0, x_j + (c_j - a_jᵀ(dual + rho_i*residual))/Σ_i rho_i*a_ij²).
                let step = (gains[j] - priced) / (rho * norms[j]);
                if !step.is_finite() || !(x[j] + step.max(-x[j])).is_finite() {
                    return Err(PyValueError::new_err("nonfinite packing coordinate update"));
                }
                let next = (x[j] + step.max(-x[j])).max(0.);
                let delta = next - x[j];
                x[j] = next;
                for (i, value) in problem.column(j) {
                    residual[i] += value * delta;
                    usage[i] += value * delta;
                }
            }
        }
        for i in 0..resource_count {
            dual[i] += rho * weights[i] * residual[i];
            if !dual[i].is_finite() || !usage[i].is_finite() {
                return Err(PyValueError::new_err("nonfinite packing resource update"));
            }
        }
        iterations = iteration + 1;
        if iteration % 10 == 0 || iterations == max_iterations {
            usage = problem.usage(&x, resource_count);
            let scale = usage.iter().copied().fold(1., f64::max);
            let gain = problem.gain(&x) / scale;
            if !scale.is_finite() || !gain.is_finite() {
                return Err(PyValueError::new_err("nonfinite packing primal bound"));
            }
            if gain > objective {
                objective = gain;
                for j in 0..candidate_count {
                    best[j] = x[j] / scale;
                }
            }
            let mut ratio = 0_f64;
            let mut covered = true;
            for j in 0..candidate_count {
                let priced: f64 = problem
                    .column(j)
                    .map(|(i, v)| v * dual[i].max(price_floor))
                    .sum();
                if !priced.is_finite() {
                    return Err(PyValueError::new_err("nonfinite packing dual coverage"));
                }
                active[j] = x[j] > 0. || priced < gains[j] * 1.01;
                if priced == 0. {
                    covered = false;
                } else {
                    ratio = ratio.max(gains[j] / priced);
                }
            }
            if covered {
                let candidate: Vec<f64> = dual
                    .iter()
                    .map(|v| v.max(price_floor) * ratio * (1. + 1e-12))
                    .collect();
                let upper: f64 = candidate.iter().sum();
                if !upper.is_finite() {
                    return Err(PyValueError::new_err("nonfinite packing dual bound"));
                }
                if upper < bound {
                    bound = upper;
                    witness = candidate;
                }
            }
        }
    }
    let mut remaining: Vec<f64> = problem
        .usage(&best, resource_count)
        .iter()
        .map(|v| (1. - v).max(0.))
        .collect();
    problem.fill(&mut best, &mut remaining)?;
    if iterations > 0 {
        problem.compress(&mut best, resource_count)?;
    }
    if best.iter().any(|v| !v.is_finite() || *v < 0.)
        || problem
            .usage(&best, resource_count)
            .iter()
            .any(|v| !v.is_finite() || *v > 1. + 1e-8)
        || witness.iter().any(|v| !v.is_finite() || *v < 0.)
        || (0..candidate_count).any(|j| {
            problem.column(j).map(|(i, v)| v * witness[i]).sum::<f64>() < gains[j] * (1. - 1e-10)
        })
        || bound < problem.gain(&best) * (1. - 1e-10)
    {
        return Err(PyValueError::new_err(
            "packing primal or dual certificate failed",
        ));
    }
    Ok((
        PyArray1::from_vec(py, best),
        PyArray1::from_vec(py, witness),
        iterations,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compression_preserves_primary_on_near_ties() {
        for scale in [1e-150, 1., 1e150] {
            for epsilon in [0., 1e-6, 1e-12, 1e-15] {
                let n = 64;
                let starts: Vec<i32> = (0..=n).map(|j| j * 6).collect();
                let rows: Vec<i32> = (0..n).flat_map(|_| 0..6).collect();
                let values: Vec<f64> = (0..n)
                    .flat_map(|j| {
                        (0..6).map(move |i| {
                            0.2 + f64::from((j % 8 + i) % 9) * 0.08 + epsilon * f64::from(j % 5)
                        })
                    })
                    .collect();
                let gains: Vec<f64> = (0..n)
                    .map(|j| scale * (1. + f64::from(j % 8) + epsilon * f64::from(j)))
                    .collect();
                let secondary: Vec<f64> = (0..n).map(|j| 1. + f64::from(j % 7)).collect();
                let problem = Packing {
                    starts: &starts,
                    rows: &rows,
                    values: &values,
                    gains: &gains,
                    secondary: &secondary,
                };
                let mut x = vec![1. / f64::from(n); n as usize];
                let gain = problem.gain(&x);
                problem.compress(&mut x, 6).unwrap();
                assert!((problem.gain(&x) - gain).abs() <= 1e-9 * gain);
                assert!(problem.usage(&x, 6).iter().all(|v| *v <= 1. + 1e-9));
                assert!(x.iter().filter(|v| **v > 0.).count() <= 7);
            }
        }
    }

    #[test]
    fn compression_batches_independent_rows_without_losing_shared_constraints() {
        let n = 1024;
        let starts: Vec<i32> = (0..=n).map(|j| j * 2).collect();
        let rows: Vec<i32> = (0..n).flat_map(|j| [0, 1 + j / 4]).collect();
        let values = vec![1.; n as usize * 2];
        let gains = vec![1.; n as usize];
        let secondary = vec![0.; n as usize];
        let problem = Packing {
            starts: &starts,
            rows: &rows,
            values: &values,
            gains: &gains,
            secondary: &secondary,
        };
        let mut x = vec![1. / f64::from(n); n as usize];
        problem.compress(&mut x, 257).unwrap();
        assert!((problem.gain(&x) - 1.).abs() < 1e-10);
        assert!(problem.usage(&x, 257).iter().all(|v| *v <= 1. + 1e-9));
        assert!(x.iter().filter(|v| **v > 0.).count() <= 6);
    }

    #[test]
    fn compression_rejects_nonfinite_normalization() {
        assert!(reduce_support(&[vec![(0, 1.)]], &[f64::INFINITY], &mut [1.], 1).is_err());
        assert!(reduce_support(
            &[vec![(0, 1.)], vec![(0, 1.)]],
            &[1., 1e-320],
            &mut [1., 1.],
            1
        )
        .is_err());
    }
}

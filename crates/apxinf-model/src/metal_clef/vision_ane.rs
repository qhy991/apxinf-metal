//! vis014 (M4) transplant sketch: a GPU/ANE row split of every vision block's
//! MLP, ApxInf side.
//!
//! **Status: skeleton, nothing here touches the ANE.** The device half (Core ML
//! program build, load gate, handoff/merge, calibration) needs the shared ANE
//! idle window and is specified in `devlocal/clef-vision/VISION-OPT-PLAN.md`.
//! The planning half below is pure, so the geometry, the per-patch-row exposure
//! budget and the ANE byte budget are pinned and unit-tested before any device
//! work starts.
//!
//! The mechanism copied from M4 `vis014-mlp-ane-advance-budget13`: each block's
//! ANE program is that block's MLP (`fc1` 1152->4304, GELU, `fc2` 4304->1152)
//! on a 512-row tile; block `b` takes the advancing row window
//! `[b*A, b*A + A) mod T'`, where `T'` is the call's patch-row count rounded
//! down to 32 rows and `A = 13 * T' / 27` rounded down to 32, so every patch row
//! spends at most 13 of its 27 MLP evaluations on the ANE and the rest on the
//! GPU. Calls under 1024 rows keep their whole MLP on the GPU (bit-identical to
//! the current path).
//!
//! Why the exposure cap is a *level* and not a per-case rule: M4 measured
//! `vis011` at mean exposure 12.8-13.1 / max 14 (16/16 quality) and two
//! 15.5-mean designs at 15/16 and 14/16, with the failures moving between cases
//! (see `notes` in that campaign). 13 keeps the mean at the proven point and
//! the max strictly below the proven one.

/// Rows of one ANE program call.
pub const ANE_TILE: usize = 512;
/// Calls smaller than this many patch rows keep the whole MLP on the GPU.
pub const SPLIT_FLOOR_ROWS: usize = 1024;
/// Per-patch-row ANE exposure cap over the 27 blocks (M4 vis014's proven 13).
pub const EXPOSURE_BUDGET: usize = 13;
/// Every window bound is a multiple of this many rows (one Metal threadgroup).
pub const ROW_ALIGNMENT: usize = 32;

/// Vision geometry the split is built for.
pub const VISION_BLOCKS: usize = 27;
pub const VISION_HIDDEN: usize = 1152;
pub const VISION_INTERMEDIATE: usize = 4304;

/// `T'`: patch rows whose ANE windows are planned; the tail (under 32 rows)
/// always stays on the GPU so a threadgroup never straddles the seam.
pub fn aligned_rows(rows: usize) -> usize {
    rows / ROW_ALIGNMENT * ROW_ALIGNMENT
}

/// `A`: rows each active block puts on the ANE, 0 when the call is not
/// eligible. `floor(exposure * T' / blocks)` rounded down to `ROW_ALIGNMENT`.
pub fn window_rows(rows: usize) -> usize {
    let aligned = aligned_rows(rows);
    if rows < SPLIT_FLOOR_ROWS || aligned == 0 {
        return 0;
    }
    let budget = EXPOSURE_BUDGET * aligned / VISION_BLOCKS;
    (budget / ROW_ALIGNMENT * ROW_ALIGNMENT).min(aligned)
}

/// One block's contiguous ANE row window, wrapping at `T'` when it runs past
/// the end (the wrap is legal because every bound is a multiple of 32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub start: usize,
    pub rows: usize,
}

impl Window {
    /// Whether patch row `row` of a `T'`-row call is inside this window.
    pub fn contains(&self, aligned: usize, row: usize) -> bool {
        if self.rows == 0 || aligned == 0 {
            return false;
        }
        let start = self.start % aligned;
        (row + aligned - start) % aligned < self.rows
    }
}

/// The split plan of one vision call, one window per block in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Patch rows of the call (the tower's sequence length).
    pub rows: usize,
    /// `T'`: rows the windows are planned over.
    pub aligned: usize,
    /// `A`: rows per active block, 0 when the call keeps every MLP on the GPU.
    pub window_rows: usize,
    /// Per block: `Some(window)` when that block's window is non-empty.
    pub windows: Vec<Option<Window>>,
}


impl Plan {
    /// Blocks whose MLP rows go to the ANE.
    pub fn active_blocks(&self) -> usize {
        self.windows.iter().flatten().count()
    }

    /// Patch rows evaluated on the ANE across all blocks.
    pub fn ane_row_slots(&self) -> usize {
        self.active_blocks() * self.window_rows
    }

    /// Largest number of blocks whose window contains one patch row.
    pub fn max_exposure(&self) -> usize {
        if self.window_rows == 0 || self.aligned == 0 {
            return 0;
        }
        (0..self.aligned)
            .map(|row| {
                self.windows
                    .iter()
                    .flatten()
                    .filter(|window| window.contains(self.aligned, row))
                    .count()
            })
            .max()
            .unwrap_or(0)
    }

    /// Fraction of the tower's MLP row evaluations that run on the ANE.
    pub fn ane_share(&self) -> f64 {
        let total = self.rows * VISION_BLOCKS;
        if total == 0 {
            return 0.0;
        }
        self.ane_row_slots() as f64 / total as f64
    }

    /// Whether this call splits at all.
    pub fn splits(&self) -> bool {
        self.window_rows > 0
    }
}

/// Plan one vision call, exactly as vis014 does: advancing window per block,
/// `A` rows each, no per-block calibration applied yet (that is the device
/// half; a calibrated block may scale `A` down or stay on the GPU).
pub fn plan(rows: usize) -> Plan {
    let aligned = aligned_rows(rows);
    let window_rows = window_rows(rows);
    let windows = (0..VISION_BLOCKS)
        .map(|block| {
            if window_rows == 0 {
                None
            } else {
                Some(Window {
                    start: (block * window_rows) % aligned,
                    rows: window_rows,
                })
            }
        })
        .collect();
    Plan {
        rows,
        aligned,
        window_rows,
        windows,
    }
}

/// Bytes of one block's ANE program: fp16 `fc1` `[intermediate, hidden]`, its
/// bias, fp16 `fc2` `[hidden, intermediate]` and its bias.
pub fn program_weight_bytes() -> usize {
    let matrix = |out: usize, input: usize| out * input * std::mem::size_of::<u16>();
    matrix(VISION_INTERMEDIATE, VISION_HIDDEN)
        + VISION_INTERMEDIATE * std::mem::size_of::<u16>()
        + matrix(VISION_HIDDEN, VISION_INTERMEDIATE)
        + VISION_HIDDEN * std::mem::size_of::<u16>()
}

/// Bytes of all 27 block programs (M4 measured 535.8 MB for the same builder).
pub fn total_weight_bytes() -> usize {
    VISION_BLOCKS * program_weight_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_rows_keeps_whole_threadgroups() {
        assert_eq!(aligned_rows(480), 480);
        assert_eq!(aligned_rows(1504), 1504);
        assert_eq!(aligned_rows(1512), 1504);
        assert_eq!(aligned_rows(1100), 1088);
        assert_eq!(aligned_rows(31), 0);
    }

    #[test]
    fn window_rows_match_the_vis014_geometry() {
        // Mario single-image crops (480 / 432 / 600 patch rows) are under the
        // floor: the whole MLP stays on the GPU, bit-identical to today.
        for rows in [0usize, 432, 480, 600, 1023] {
            assert_eq!(window_rows(rows), 0, "{rows}");
        }
        assert_eq!(window_rows(1024), 480);
        assert_eq!(window_rows(1504), 704);
        assert_eq!(window_rows(1512), 704);
        // The M4 cell sizes, for cross-checking against its published table.
        assert_eq!(window_rows(1920), 896);
        assert_eq!(window_rows(4320), 2080);
        assert_eq!(window_rows(6336), 3040);
    }

    #[test]
    fn every_patch_row_is_exposed_at_most_thirteen_times() {
        for rows in [1024usize, 1512, 1920, 4320, 6336] {
            let plan = plan(rows);
            assert!(plan.splits(), "{rows}");
            let exposure = plan.max_exposure();
            let share = EXPOSURE_BUDGET * plan.aligned / VISION_BLOCKS;
            assert!(
                exposure <= EXPOSURE_BUDGET,
                "rows {rows}: max exposure {exposure} exceeds {EXPOSURE_BUDGET}"
            );
            // The advancing window spends the budget evenly: every row is in
            // either floor or ceil(blocks * A / T') windows.
            let mean = VISION_BLOCKS * plan.window_rows / plan.aligned;
            assert!(exposure <= mean + 1, "rows {rows}: uneven windows");
            assert!(plan.ane_row_slots() <= VISION_BLOCKS * share + VISION_BLOCKS * ROW_ALIGNMENT);
            // Every active window is 32-row aligned and fits the ring.
            for window in plan.windows.iter().flatten() {
                assert_eq!(window.rows % ROW_ALIGNMENT, 0);
                assert_eq!(window.start % ROW_ALIGNMENT, 0);
                assert!(window.start < plan.aligned);
                assert!(window.rows <= plan.aligned);
            }
            assert_eq!(plan.ane_row_slots(), plan.active_blocks() * plan.window_rows);
        }
    }

    #[test]
    fn under_the_floor_nothing_is_split() {
        let plan = plan(480);
        assert!(!plan.splits());
        assert_eq!(plan.active_blocks(), 0);
        assert_eq!(plan.ane_share(), 0.0);
        assert!(plan.windows.iter().all(Option::is_none));
    }

    #[test]
    fn three_image_requests_split_about_47_percent_of_the_mlp_rows() {
        // The joint Mario protocol: 480 + 432 + 600 = 1512 patch rows.
        let plan = plan(1512);
        assert_eq!(plan.aligned, 1504); // 8-row tail stays on the GPU
        assert_eq!(plan.window_rows, 704);
        assert_eq!(plan.active_blocks(), VISION_BLOCKS);
        assert_eq!(plan.ane_row_slots(), 27 * 704);
        let share = plan.ane_share();
        assert!((share - 0.468).abs() < 0.005, "share {share}");
        assert!(plan.max_exposure() <= EXPOSURE_BUDGET);
    }

    #[test]
    fn ane_weight_bytes_match_the_m4_builder() {
        // 19.83 MB per block, 535.8 MB for the 27 blocks (M4 vis014 numbers).
        assert_eq!(program_weight_bytes(), 19_843_744);
        assert_eq!(total_weight_bytes(), 535_781_088);
    }
}

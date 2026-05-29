//! Input component for tracking input state.

use crate::components::{Component, StorageType};

/// InputState component for tracking entity-specific input bindings.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct InputState {
    pub move_forward: bool,
    pub move_backward: bool,
    pub move_left: bool,
    pub move_right: bool,
    pub jump: bool,
    pub sprint: bool,
    pub interact: bool,
}

impl Component for InputState {
    const STORAGE_TYPE: StorageType = StorageType::SparseSet;
}

impl InputState {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn any_movement(&self) -> bool {
        self.move_forward || self.move_backward || self.move_left || self.move_right
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_state_default() {
        let input = InputState::default();
        assert!(!input.jump);
        assert!(!input.any_movement());
    }

    #[test]
    fn test_any_movement() {
        let mut input = InputState::default();
        input.move_forward = true;
        assert!(input.any_movement());
    }
}

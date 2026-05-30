// src/undo_stack.rs - Undo/redo system for editor and tools
use std::collections::VecDeque;
use parking_lot::RwLock;

pub trait UndoAction: Send + Sync {
    fn undo(&self);
    fn redo(&self);
    fn merge_with(&self, _other: &dyn UndoAction) -> Option<Box<dyn UndoAction>> { None }
}

pub struct UndoStack {
    undo_stack: RwLock<VecDeque<Box<dyn UndoAction>>>,
    redo_stack: RwLock<VecDeque<Box<dyn UndoAction>>>,
    max_size: usize,
}

impl UndoStack {
    pub fn new(max_size: usize) -> Self { Self { undo_stack: RwLock::new(VecDeque::with_capacity(max_size)), redo_stack: RwLock::new(VecDeque::new()), max_size } }
    pub fn push(&self, action: Box<dyn UndoAction>) {
        let mut stack = self.undo_stack.write();
        if stack.len() >= self.max_size { stack.pop_front(); }
        stack.push_back(action);
        self.redo_stack.write().clear();
    }
    pub fn undo(&self) {
        if let Some(action) = self.undo_stack.write().pop_back() {
            action.undo();
            self.redo_stack.write().push_back(action);
        }
    }
    pub fn redo(&self) {
        if let Some(action) = self.redo_stack.write().pop_back() {
            action.redo();
            self.undo_stack.write().push_back(action);
        }
    }
}

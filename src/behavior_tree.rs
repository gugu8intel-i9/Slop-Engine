// src/behavior_tree.rs
// AI logic nodes, selectors, sequences, decorators

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

/// Behavior tree execution status
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NodeStatus {
    Success,
    Failure,
    Running,
}

/// Blackboard for sharing data between nodes
#[derive(Clone, Debug)]
pub struct Blackboard {
    pub data: HashMap<String, BlackboardValue>,
}

#[derive(Clone, Debug)]
pub enum BlackboardValue {
    Bool(bool),
    Int(i32),
    Float(f32),
    String(String),
    Vector(glam::Vec3),
    Entity(u64),
}

impl Blackboard {
    pub fn new() -> Self {
        Self { data: HashMap::new() }
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.data.get(key).and_then(|v| match v {
            BlackboardValue::Bool(b) => Some(*b),
            _ => None,
        })
    }

    pub fn set_bool(&mut self, key: &str, value: bool) {
        self.data.insert(key.to_string(), BlackboardValue::Bool(value));
    }

    pub fn get_float(&self, key: &str) -> Option<f32> {
        self.data.get(key).and_then(|v| match v {
            BlackboardValue::Float(f) => Some(*f),
            _ => None,
        })
    }

    pub fn set_float(&mut self, key: &str, value: f32) {
        self.data.insert(key.to_string(), BlackboardValue::Float(value));
    }

    pub fn get_int(&self, key: &str) -> Option<i32> {
        self.data.get(key).and_then(|v| match v {
            BlackboardValue::Int(i) => Some(*i),
            _ => None,
        })
    }

    pub fn set_int(&mut self, key: &str, value: i32) {
        self.data.insert(key.to_string(), BlackboardValue::Int(value));
    }

    pub fn get_vector(&self, key: &str) -> Option<glam::Vec3> {
        self.data.get(key).and_then(|v| match v {
            BlackboardValue::Vector(v) => Some(*v),
            _ => None,
        })
    }

    pub fn set_vector(&mut self, key: &str, value: glam::Vec3) {
        self.data.insert(key.to_string(), BlackboardValue::Vector(value));
    }

    pub fn has(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    pub fn remove(&mut self, key: &str) {
        self.data.remove(key);
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }
}

impl Default for Blackboard {
    fn default() -> Self {
        Self::new()
    }
}

/// Behavior tree node trait
pub trait BehaviorNode: Send + Sync {
    fn execute(&self, blackboard: &Blackboard) -> NodeStatus;
    fn name(&self) -> &str;
    fn clone_box(&self) -> Box<dyn BehaviorNode>;
}

/// Composite node base
#[derive(Clone)]
pub struct CompositeNode {
    pub name: String,
    pub children: Vec<Box<dyn BehaviorNode>>,
    pub current_child: usize,
}

impl CompositeNode {
    pub fn new(name: String) -> Self {
        Self {
            name,
            children: Vec::new(),
            current_child: 0,
        }
    }

    pub fn add_child(&mut self, child: Box<dyn BehaviorNode>) {
        self.children.push(child);
    }
}

/// Selector node - returns success if any child succeeds
pub struct Selector {
    inner: RwLock<CompositeNode>,
}

impl Selector {
    pub fn new(name: &str) -> Self {
        Self {
            inner: RwLock::new(CompositeNode::new(name.to_string())),
        }
    }

    pub fn add_child(&self, child: Box<dyn BehaviorNode>) {
        self.inner.write().children.push(child);
    }
}

impl BehaviorNode for Selector {
    fn execute(&self, blackboard: &Blackboard) -> NodeStatus {
        let mut inner = self.inner.write();
        
        for i in inner.current_child..inner.children.len() {
            inner.current_child = i;
            let status = inner.children[i].execute(blackboard);
            
            match status {
                NodeStatus::Success => {
                    inner.current_child = 0;
                    return NodeStatus::Success;
                }
                NodeStatus::Running => return NodeStatus::Running,
                NodeStatus::Failure => continue,
            }
        }
        
        inner.current_child = 0;
        NodeStatus::Failure
    }

    fn name(&self) -> &str {
        &self.inner.read().name
    }

    fn clone_box(&self) -> Box<dyn BehaviorNode> {
        Box::new(self.clone())
    }
}

impl Clone for Selector {
    fn clone(&self) -> Self {
        Self {
            inner: RwLock::new(self.inner.read().clone()),
        }
    }
}

/// Sequence node - returns success only if all children succeed
pub struct Sequence {
    inner: RwLock<CompositeNode>,
}

impl Sequence {
    pub fn new(name: &str) -> Self {
        Self {
            inner: RwLock::new(CompositeNode::new(name.to_string())),
        }
    }

    pub fn add_child(&self, child: Box<dyn BehaviorNode>) {
        self.inner.write().children.push(child);
    }
}

impl BehaviorNode for Sequence {
    fn execute(&self, blackboard: &Blackboard) -> NodeStatus {
        let mut inner = self.inner.write();
        
        for i in inner.current_child..inner.children.len() {
            inner.current_child = i;
            let status = inner.children[i].execute(blackboard);
            
            match status {
                NodeStatus::Success => continue,
                NodeStatus::Running => return NodeStatus::Running,
                NodeStatus::Failure => {
                    inner.current_child = 0;
                    return NodeStatus::Failure;
                }
            }
        }
        
        inner.current_child = 0;
        NodeStatus::Success
    }

    fn name(&self) -> &str {
        &self.inner.read().name
    }

    fn clone_box(&self) -> Box<dyn BehaviorNode> {
        Box::new(self.clone())
    }
}

impl Clone for Sequence {
    fn clone(&self) -> Self {
        Self {
            inner: RwLock::new(self.inner.read().clone()),
        }
    }
}

/// Parallel node - executes multiple children simultaneously
pub struct Parallel {
    inner: RwLock<CompositeNode>,
    required_successes: usize,
}

impl Parallel {
    pub fn new(name: &str, required_successes: usize) -> Self {
        Self {
            inner: RwLock::new(CompositeNode::new(name.to_string())),
            required_successes,
        }
    }

    pub fn add_child(&self, child: Box<dyn BehaviorNode>) {
        self.inner.write().children.push(child);
    }
}

impl BehaviorNode for Parallel {
    fn execute(&self, blackboard: &Blackboard) -> NodeStatus {
        let inner = self.inner.read();
        let mut successes = 0;
        let mut running = false;
        
        for child in &inner.children {
            match child.execute(blackboard) {
                NodeStatus::Success => successes += 1,
                NodeStatus::Running => running = true,
                NodeStatus::Failure => {}
            }
        }
        
        if successes >= self.required_successes {
            NodeStatus::Success
        } else if running {
            NodeStatus::Running
        } else {
            NodeStatus::Failure
        }
    }

    fn name(&self) -> &str {
        &self.inner.read().name
    }

    fn clone_box(&self) -> Box<dyn BehaviorNode> {
        Box::new(self.clone())
    }
}

impl Clone for Parallel {
    fn clone(&self) -> Self {
        Self {
            inner: RwLock::new(self.inner.read().clone()),
            required_successes: self.required_successes,
        }
    }
}

/// Decorator node - modifies behavior of a single child
pub struct Decorator {
    pub name: String,
    pub child: Option<Box<dyn BehaviorNode>>,
    pub decorator_type: DecoratorType,
}

#[derive(Clone)]
pub enum DecoratorType {
    Invert,
    Repeat(usize),
    SucceedAlways,
    FailAlways,
    LimitTime(f32),
    Cooldown(f32),
    Condition(Box<dyn Fn(&Blackboard) -> bool + Send + Sync>),
}

impl Decorator {
    pub fn new(name: &str, decorator_type: DecoratorType) -> Self {
        Self {
            name: name.to_string(),
            child: None,
            decorator_type,
        }
    }

    pub fn set_child(&mut self, child: Box<dyn BehaviorNode>) {
        self.child = Some(child);
    }
}

impl BehaviorNode for Decorator {
    fn execute(&self, blackboard: &Blackboard) -> NodeStatus {
        let child = match &self.child {
            Some(c) => c,
            None => return NodeStatus::Failure,
        };

        match &self.decorator_type {
            DecoratorType::Invert => {
                match child.execute(blackboard) {
                    NodeStatus::Success => NodeStatus::Failure,
                    NodeStatus::Failure => NodeStatus::Success,
                    NodeStatus::Running => NodeStatus::Running,
                }
            }
            DecoratorType::Repeat(max_repeats) => {
                // Simplified repeat - would need state tracking for proper implementation
                let result = child.execute(blackboard);
                if result == NodeStatus::Success && *max_repeats > 0 {
                    NodeStatus::Running // Continue repeating
                } else {
                    result
                }
            }
            DecoratorType::SucceedAlways => {
                match child.execute(blackboard) {
                    NodeStatus::Running => NodeStatus::Running,
                    _ => NodeStatus::Success,
                }
            }
            DecoratorType::FailAlways => {
                match child.execute(blackboard) {
                    NodeStatus::Running => NodeStatus::Running,
                    _ => NodeStatus::Failure,
                }
            }
            DecoratorType::LimitTime(max_time) => {
                // Would need timestamp tracking for proper implementation
                child.execute(blackboard)
            }
            DecoratorType::Cooldown(cooldown_time) => {
                // Would need cooldown tracking for proper implementation
                child.execute(blackboard)
            }
            DecoratorType::Condition(condition_fn) => {
                if condition_fn(blackboard) {
                    child.execute(blackboard)
                } else {
                    NodeStatus::Failure
                }
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn clone_box(&self) -> Box<dyn BehaviorNode> {
        Box::new(self.clone())
    }
}

impl Clone for Decorator {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            child: self.child.as_ref().map(|c| c.clone_box()),
            decorator_type: self.decorator_type.clone(),
        }
    }
}

/// Action node - leaf node that performs an action
pub struct Action {
    pub name: String,
    pub action_fn: Arc<dyn Fn(&Blackboard) -> NodeStatus + Send + Sync>,
}

impl Action {
    pub fn new<F>(name: &str, action_fn: F) -> Self
    where
        F: Fn(&Blackboard) -> NodeStatus + Send + Sync + 'static,
    {
        Self {
            name: name.to_string(),
            action_fn: Arc::new(action_fn),
        }
    }
}

impl BehaviorNode for Action {
    fn execute(&self, blackboard: &Blackboard) -> NodeStatus {
        (self.action_fn)(blackboard)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn clone_box(&self) -> Box<dyn BehaviorNode> {
        Box::new(self.clone())
    }
}

impl Clone for Action {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            action_fn: self.action_fn.clone(),
        }
    }
}

/// Condition node - leaf node that checks a condition
pub struct Condition {
    pub name: String,
    pub condition_fn: Arc<dyn Fn(&Blackboard) -> bool + Send + Sync>,
}

impl Condition {
    pub fn new<F>(name: &str, condition_fn: F) -> Self
    where
        F: Fn(&Blackboard) -> bool + Send + Sync + 'static,
    {
        Self {
            name: name.to_string(),
            condition_fn: Arc::new(condition_fn),
        }
    }
}

impl BehaviorNode for Condition {
    fn execute(&self, blackboard: &Blackboard) -> NodeStatus {
        if (self.condition_fn)(blackboard) {
            NodeStatus::Success
        } else {
            NodeStatus::Failure
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn clone_box(&self) -> Box<dyn BehaviorNode> {
        Box::new(self.clone())
    }
}

impl Clone for Condition {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            condition_fn: self.condition_fn.clone(),
        }
    }
}

/// Wait node - delays execution for a specified time
pub struct Wait {
    pub name: String,
    pub duration: f32,
}

impl Wait {
    pub fn new(name: &str, duration: f32) -> Self {
        Self {
            name: name.to_string(),
            duration,
        }
    }
}

impl BehaviorNode for Wait {
    fn execute(&self, _blackboard: &Blackboard) -> NodeStatus {
        // Would need timing system integration for proper implementation
        NodeStatus::Running
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn clone_box(&self) -> Box<dyn BehaviorNode> {
        Box::new(self.clone())
    }
}

impl Clone for Wait {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            duration: self.duration,
        }
    }
}

/// Complete behavior tree
pub struct BehaviorTree {
    pub root: Box<dyn BehaviorNode>,
    pub blackboard: RwLock<Blackboard>,
    pub is_running: bool,
    pub debug_info: RwLock<String>,
}

impl BehaviorTree {
    pub fn new(root: Box<dyn BehaviorNode>) -> Self {
        Self {
            root,
            blackboard: RwLock::new(Blackboard::new()),
            is_running: false,
            debug_info: RwLock::new(String::new()),
        }
    }

    pub fn update(&self) -> NodeStatus {
        let blackboard = self.blackboard.read();
        let status = self.root.execute(&blackboard);
        
        self.is_running = status == NodeStatus::Running;
        
        drop(blackboard);
        
        if cfg!(debug_assertions) {
            let mut debug = self.debug_info.write();
            *debug = format!("Last status: {:?}", status);
        }
        
        status
    }

    pub fn stop(&self) {
        self.is_running = false;
    }

    pub fn reset(&self) {
        self.blackboard.write().clear();
        self.is_running = false;
    }

    pub fn get_blackboard(&self) -> std::sync::MutexGuard<Blackboard> {
        // Using parking_lot's mapped guards would be better here
        unimplemented!("Use direct blackboard access methods")
    }

    pub fn set_value(&self, key: &str, value: BlackboardValue) {
        self.blackboard.write().data.insert(key.to_string(), value);
    }

    pub fn get_value(&self, key: &str) -> Option<BlackboardValue> {
        self.blackboard.read().data.get(key).cloned()
    }
}

/// Builder for creating behavior trees fluently
pub struct BehaviorTreeBuilder {
    name: String,
}

impl BehaviorTreeBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }

    pub fn selector() -> Selector {
        Selector::new("Selector")
    }

    pub fn sequence() -> Sequence {
        Sequence::new("Sequence")
    }

    pub fn parallel(required: usize) -> Parallel {
        Parallel::new("Parallel", required)
    }

    pub fn action<F>(name: &str, f: F) -> Action
    where
        F: Fn(&Blackboard) -> NodeStatus + Send + Sync + 'static,
    {
        Action::new(name, f)
    }

    pub fn condition<F>(name: &str, f: F) -> Condition
    where
        F: Fn(&Blackboard) -> bool + Send + Sync + 'static,
    {
        Condition::new(name, f)
    }

    pub fn invert(child: Box<dyn BehaviorNode>) -> Decorator {
        let mut dec = Decorator::new("Invert", DecoratorType::Invert);
        dec.set_child(child);
        dec
    }

    pub fn succeed_always(child: Box<dyn BehaviorNode>) -> Decorator {
        let mut dec = Decorator::new("SucceedAlways", DecoratorType::SucceedAlways);
        dec.set_child(child);
        dec
    }

    pub fn fail_always(child: Box<dyn BehaviorNode>) -> Decorator {
        let mut dec = Decorator::new("FailAlways", DecoratorType::FailAlways);
        dec.set_child(child);
        dec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selector_success() {
        let selector = Selector::new("TestSelector");
        selector.add_child(Box::new(Action::new("FailAction", |_| NodeStatus::Failure)));
        selector.add_child(Box::new(Action::new("SuccessAction", |_| NodeStatus::Success)));

        let bb = Blackboard::new();
        assert_eq!(selector.execute(&bb), NodeStatus::Success);
    }

    #[test]
    fn test_sequence_success() {
        let sequence = Sequence::new("TestSequence");
        sequence.add_child(Box::new(Action::new("Success1", |_| NodeStatus::Success)));
        sequence.add_child(Box::new(Action::new("Success2", |_| NodeStatus::Success)));

        let bb = Blackboard::new();
        assert_eq!(sequence.execute(&bb), NodeStatus::Success);
    }

    #[test]
    fn test_sequence_failure() {
        let sequence = Sequence::new("TestSequence");
        sequence.add_child(Box::new(Action::new("Success1", |_| NodeStatus::Success)));
        sequence.add_child(Box::new(Action::new("Fail", |_| NodeStatus::Failure)));
        sequence.add_child(Box::new(Action::new("Success2", |_| NodeStatus::Success)));

        let bb = Blackboard::new();
        assert_eq!(sequence.execute(&bb), NodeStatus::Failure);
    }

    #[test]
    fn test_invert_decorator() {
        let mut invert = Decorator::new("Invert", DecoratorType::Invert);
        invert.set_child(Box::new(Action::new("Fail", |_| NodeStatus::Failure)));

        let bb = Blackboard::new();
        assert_eq!(invert.execute(&bb), NodeStatus::Success);
    }

    #[test]
    fn test_blackboard() {
        let mut bb = Blackboard::new();
        bb.set_float("health", 50.0);
        bb.set_bool("is_alive", true);

        assert_eq!(bb.get_float("health"), Some(50.0));
        assert_eq!(bb.get_bool("is_alive"), Some(true));
    }
}

// src/editor.rs - In-engine editor mode, gizmos, inspector, scene view
use glam::{Vec3, Mat4};
use parking_lot::RwLock;

pub struct Editor {
    is_active: bool,
    selected_entity: Option<u64>,
    gizmo_mode: GizmoMode,
}

#[derive(Clone, Copy, Debug)]
pub enum GizmoMode { Translate, Rotate, Scale }

impl Editor {
    pub fn new() -> Self { Self { is_active: false, selected_entity: None, gizmo_mode: GizmoMode::Translate } }
    pub fn toggle(&mut self) { self.is_active = !self.is_active; }
    pub fn select(&mut self, entity: u64) { self.selected_entity = Some(entity); }
}

pub struct Inspector { properties: Vec<Property> }
pub struct Property { name: String, value: PropertyValue }
#[derive(Clone)] pub enum PropertyValue { Float(f32), Int(i32), String(String), Bool(bool), Vec3(Vec3) }

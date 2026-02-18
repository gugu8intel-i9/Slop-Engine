// src/input_system.rs
// High-performance Input System for games and engines.
// Supports winit events (native) and web bindings (wasm).
// No external crates required beyond winit and optionally web-sys for wasm.

use std::collections::HashMap;
use std::time::Instant;
use winit::event::{ElementState, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent};

// Compact handle types for actions and axes
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct ActionId(pub u32);

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct AxisId(pub u32);

// Physical input enums
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Key {
    Code(VirtualKeyCode),
    Scancode(u32),
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum MouseBtn {
    Left,
    Right,
    Middle,
    Other(u8),
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum GamepadBtn {
    Button(u8),
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum GamepadAxis {
    Axis(u8),
}

// Internal per-button state (small POD)
#[derive(Copy, Clone, Debug)]
struct ButtonState {
    down: bool,
    last_down_frame: u64,
    last_up_frame: u64,
}

impl ButtonState {
    fn new() -> Self { Self { down: false, last_down_frame: 0, last_up_frame: 0 } }
}

// Ring buffer for events and text input
struct RingBuffer<T> {
    buf: Vec<Option<T>>,
    head: usize,
    tail: usize,
    cap: usize,
}

impl<T> RingBuffer<T> {
    fn with_capacity(cap: usize) -> Self {
        let mut buf = Vec::with_capacity(cap);
        buf.resize_with(cap, || None);
        Self { buf, head: 0, tail: 0, cap }
    }
    fn push(&mut self, v: T) -> bool {
        let next = (self.tail + 1) % self.cap;
        if next == self.head { return false; } // full
        self.buf[self.tail] = Some(v);
        self.tail = next;
        true
    }
    fn pop(&mut self) -> Option<T> {
        if self.head == self.tail { return None; }
        let v = self.buf[self.head].take();
        self.head = (self.head + 1) % self.cap;
        v
    }
    fn clear(&mut self) {
        while self.pop().is_some() {}
    }
}

// Action binding types
#[derive(Clone, Debug)]
pub enum Binding {
    Key(Key),
    Mouse(MouseBtn),
    GamepadButton { pad_id: u32, btn: GamepadBtn },
    GamepadAxis { pad_id: u32, axis: GamepadAxis, scale: f32 },
    CompositeAxis { negative: Key, positive: Key }, // simple keyboard axis
}

// Axis binding supports multiple bindings
#[derive(Clone, Debug)]
pub struct AxisBinding {
    pub bindings: Vec<Binding>,
    pub deadzone: f32,
    pub sensitivity: f32,
    pub smoothing: f32, // 0 = none, >0 smoothing factor
}

// Main InputManager
pub struct InputManager {
    // frame counter for edge detection
    frame: u64,
    last_update: Instant,

    // raw states
    key_states: HashMap<Key, ButtonState>,
    mouse_states: HashMap<MouseBtn, ButtonState>,
    // gamepad states: map pad_id -> button/axis states
    gamepad_buttons: HashMap<(u32, u8), ButtonState>,
    gamepad_axes: HashMap<(u32, u8), f32>,

    // pointer
    pointer_pos: (f32, f32),
    pointer_prev: (f32, f32),
    pointer_delta: (f32, f32),
    scroll_delta: (f32, f32),

    // action map
    action_map: HashMap<ActionId, Vec<Binding>>,
    axis_map: HashMap<AxisId, AxisBinding>,

    // action cached states (derived each frame)
    action_down_cache: HashMap<ActionId, bool>,
    action_pressed_cache: HashMap<ActionId, bool>,
    action_released_cache: HashMap<ActionId, bool>,

    // axis cached values
    axis_cache: HashMap<AxisId, f32>,
    axis_smooth_cache: HashMap<AxisId, f32>,

    // text input queue (bounded)
    text_queue: RingBuffer<String>,

    // event queue for external ingestion (bounded)
    event_queue: RingBuffer<winit::event::WindowEvent<'static>>,
}

impl InputManager {
    pub fn new() -> Self {
        Self {
            frame: 0,
            last_update: Instant::now(),
            key_states: HashMap::with_capacity(256),
            mouse_states: HashMap::with_capacity(16),
            gamepad_buttons: HashMap::with_capacity(64),
            gamepad_axes: HashMap::with_capacity(64),
            pointer_pos: (0.0, 0.0),
            pointer_prev: (0.0, 0.0),
            pointer_delta: (0.0, 0.0),
            scroll_delta: (0.0, 0.0),
            action_map: HashMap::new(),
            axis_map: HashMap::new(),
            action_down_cache: HashMap::new(),
            action_pressed_cache: HashMap::new(),
            action_released_cache: HashMap::new(),
            axis_cache: HashMap::new(),
            axis_smooth_cache: HashMap::new(),
            text_queue: RingBuffer::with_capacity(64),
            event_queue: RingBuffer::with_capacity(512),
        }
    }

    // ---------------- Event ingestion ----------------
    // Call from winit event loop. We copy minimal data into our bounded queue.
    pub fn ingest_winit_event(&mut self, event: &winit::event::Event<()>) {
        // Only accept WindowEvent variants; convert to owned WindowEvent<'static>
        if let winit::event::Event::WindowEvent { event: we, .. } = event {
            // Convert to owned by cloning necessary parts
            // We only store a subset to keep memory small; store full WindowEvent if needed.
            let owned = match we {
                WindowEvent::KeyboardInput { device_id: _, input, is_synthetic } => {
                    // clone keyboard input with static lifetime
                    let ki = KeyboardInput {
                        scancode: input.scancode,
                        state: input.state,
                        virtual_keycode: input.virtual_keycode,
                        modifiers: input.modifiers,
                    };
                    WindowEvent::KeyboardInput { device_id: winit::event::DeviceId::dummy(), input: ki, is_synthetic: *is_synthetic }
                }
                WindowEvent::CursorMoved { device_id: _, position, modifiers: _ } => {
                    WindowEvent::CursorMoved { device_id: winit::event::DeviceId::dummy(), position: *position, modifiers: winit::event::ModifiersState::empty() }
                }
                WindowEvent::MouseInput { device_id: _, state, button, modifiers: _ } => {
                    WindowEvent::MouseInput { device_id: winit::event::DeviceId::dummy(), state: *state, button: *button, modifiers: winit::event::ModifiersState::empty() }
                }
                WindowEvent::MouseWheel { device_id: _, delta, modifiers: _ } => {
                    WindowEvent::MouseWheel { device_id: winit::event::DeviceId::dummy(), delta: *delta, modifiers: winit::event::ModifiersState::empty() }
                }
                WindowEvent::ReceivedCharacter(c) => WindowEvent::ReceivedCharacter(*c),
                _ => return, // ignore other events in ingestion path
            };
            // push into ring buffer; drop if full
            let _ = self.event_queue.push(owned);
        }
    }

    // ---------------- Low-level raw updates ----------------
    // These update raw states immediately and are used by update() to compute derived states.
    pub fn raw_key_event(&mut self, key: Key, state: ElementState) {
        let entry = self.key_states.entry(key).or_insert_with(ButtonState::new);
        match state {
            ElementState::Pressed => {
                if !entry.down {
                    entry.down = true;
                    entry.last_down_frame = self.frame + 1; // will be processed next update
                }
            }
            ElementState::Released => {
                if entry.down {
                    entry.down = false;
                    entry.last_up_frame = self.frame + 1;
                }
            }
        }
    }

    pub fn raw_mouse_event(&mut self, btn: MouseBtn, state: ElementState) {
        let entry = self.mouse_states.entry(btn).or_insert_with(ButtonState::new);
        match state {
            ElementState::Pressed => {
                if !entry.down {
                    entry.down = true;
                    entry.last_down_frame = self.frame + 1;
                }
            }
            ElementState::Released => {
                if entry.down {
                    entry.down = false;
                    entry.last_up_frame = self.frame + 1;
                }
            }
        }
    }

    pub fn raw_pointer_move(&mut self, x: f32, y: f32) {
        self.pointer_pos = (x, y);
    }

    pub fn raw_scroll(&mut self, dx: f32, dy: f32) {
        self.scroll_delta.0 += dx;
        self.scroll_delta.1 += dy;
    }

    // Gamepad raw updates (caller polls gamepad API and calls these)
    pub fn raw_gamepad_button(&mut self, pad_id: u32, btn: u8, pressed: bool) {
        let key = (pad_id, btn);
        let entry = self.gamepad_buttons.entry(key).or_insert_with(ButtonState::new);
        if pressed {
            if !entry.down {
                entry.down = true;
                entry.last_down_frame = self.frame + 1;
            }
        } else {
            if entry.down {
                entry.down = false;
                entry.last_up_frame = self.frame + 1;
            }
        }
    }

    pub fn raw_gamepad_axis(&mut self, pad_id: u32, axis: u8, value: f32) {
        self.gamepad_axes.insert((pad_id, axis), value);
    }

    // ---------------- Action and Axis binding API ----------------
    pub fn create_action(&mut self, id: ActionId) {
        self.action_map.entry(id).or_insert_with(Vec::new);
    }

    pub fn bind_action(&mut self, id: ActionId, binding: Binding) {
        self.action_map.entry(id).or_insert_with(Vec::new).push(binding);
    }

    pub fn unbind_action(&mut self, id: ActionId) {
        self.action_map.remove(&id);
    }

    pub fn create_axis(&mut self, id: AxisId, binding: AxisBinding) {
        self.axis_map.insert(id, binding);
    }

    pub fn rebind_axis(&mut self, id: AxisId, binding: AxisBinding) {
        self.axis_map.insert(id, binding);
    }

    // ---------------- Per-frame update ----------------
    // Call once per frame. dt in seconds.
    pub fn update(&mut self, dt: f32) {
        self.frame = self.frame.wrapping_add(1);
        let now = Instant::now();
        let _elapsed = now.duration_since(self.last_update);
        self.last_update = now;

        // process queued window events (bounded)
        while let Some(ev) = self.event_queue.pop() {
            match ev {
                WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(vk) = input.virtual_keycode {
                        self.raw_key_event(Key::Code(vk), input.state);
                    } else {
                        self.raw_key_event(Key::Scancode(input.scancode), input.state);
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    self.raw_pointer_move(position.x as f32, position.y as f32);
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    let mb = match button {
                        winit::event::MouseButton::Left => MouseBtn::Left,
                        winit::event::MouseButton::Right => MouseBtn::Right,
                        winit::event::MouseButton::Middle => MouseBtn::Middle,
                        winit::event::MouseButton::Other(b) => MouseBtn::Other(b),
                    };
                    self.raw_mouse_event(mb, state);
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    match delta {
                        winit::event::MouseScrollDelta::LineDelta(x, y) => self.raw_scroll(x, y),
                        winit::event::MouseScrollDelta::PixelDelta(p) => self.raw_scroll(p.x as f32, p.y as f32),
                    }
                }
                WindowEvent::ReceivedCharacter(c) => {
                    // push to text queue; drop if full
                    let mut s = String::new();
                    s.push(c);
                    let _ = self.text_queue.push(s);
                }
                _ => {}
            }
        }

        // pointer delta
        self.pointer_delta = (self.pointer_pos.0 - self.pointer_prev.0, self.pointer_pos.1 - self.pointer_prev.1);
        self.pointer_prev = self.pointer_pos;

        // compute action caches
        self.action_down_cache.clear();
        self.action_pressed_cache.clear();
        self.action_released_cache.clear();

        for (aid, bindings) in &self.action_map {
            let mut down = false;
            let mut pressed = false;
            let mut released = false;
            for b in bindings.iter() {
                match b {
                    Binding::Key(k) => {
                        if let Some(st) = self.key_states.get(k) {
                            if st.down { down = true; }
                            if st.last_down_frame == self.frame { pressed = true; }
                            if st.last_up_frame == self.frame { released = true; }
                        }
                    }
                    Binding::Mouse(mb) => {
                        if let Some(st) = self.mouse_states.get(mb) {
                            if st.down { down = true; }
                            if st.last_down_frame == self.frame { pressed = true; }
                            if st.last_up_frame == self.frame { released = true; }
                        }
                    }
                    Binding::GamepadButton { pad_id, btn } => {
                        let key = (*pad_id, match btn { GamepadBtn::Button(b) => *b });
                        if let Some(st) = self.gamepad_buttons.get(&key) {
                            if st.down { down = true; }
                            if st.last_down_frame == self.frame { pressed = true; }
                            if st.last_up_frame == self.frame { released = true; }
                        }
                    }
                    Binding::GamepadAxis { pad_id, axis, scale } => {
                        let key = (*pad_id, match axis { GamepadAxis::Axis(a) => *a });
                        if let Some(v) = self.gamepad_axes.get(&key) {
                            if v.abs() > 0.0001 { down = true; }
                            // axis bindings don't produce pressed/released edges here
                        }
                    }
                    Binding::CompositeAxis { negative, positive } => {
                        if let Some(st) = self.key_states.get(negative) {
                            if st.down { down = true; }
                            if st.last_down_frame == self.frame { pressed = true; }
                            if st.last_up_frame == self.frame { released = true; }
                        }
                        if let Some(st) = self.key_states.get(positive) {
                            if st.down { down = true; }
                            if st.last_down_frame == self.frame { pressed = true; }
                            if st.last_up_frame == self.frame { released = true; }
                        }
                    }
                }
            }
            self.action_down_cache.insert(*aid, down);
            self.action_pressed_cache.insert(*aid, pressed);
            self.action_released_cache.insert(*aid, released);
        }

        // compute axes
        for (axis_id, binding) in &self.axis_map {
            let mut value: f32 = 0.0;
            for b in &binding.bindings {
                match b {
                    Binding::Key(Key::Code(k)) => {
                        if let Some(st) = self.key_states.get(&Key::Code(*k)) {
                            if st.down { value += 1.0; }
                        }
                    }
                    Binding::Key(Key::Scancode(sc)) => {
                        if let Some(st) = self.key_states.get(&Key::Scancode(*sc)) {
                            if st.down { value += 1.0; }
                        }
                    }
                    Binding::GamepadAxis { pad_id, axis, scale } => {
                        let key = (*pad_id, match axis { GamepadAxis::Axis(a) => *a });
                        if let Some(v) = self.gamepad_axes.get(&key) {
                            value += v * (*scale);
                        }
                    }
                    Binding::CompositeAxis { negative, positive } => {
                        let neg = self.key_states.get(negative).map(|s| if s.down { 1.0 } else { 0.0 }).unwrap_or(0.0);
                        let pos = self.key_states.get(positive).map(|s| if s.down { 1.0 } else { 0.0 }).unwrap_or(0.0);
                        value += pos - neg;
                    }
                    _ => {}
                }
            }
            // apply deadzone
            if value.abs() < binding.deadzone { value = 0.0; }
            // sensitivity
            value *= binding.sensitivity;
            // smoothing (exponential)
            let prev = *self.axis_smooth_cache.get(axis_id).unwrap_or(&0.0);
            let smooth = binding.smoothing;
            let out = if smooth > 0.0 {
                prev + (value - prev) * (1.0 - (-smooth * dt).exp())
            } else {
                value
            };
            self.axis_smooth_cache.insert(*axis_id, out);
            self.axis_cache.insert(*axis_id, out);
        }

        // reset scroll delta after frame
        self.scroll_delta = (0.0, 0.0);
    }

    // ---------------- Query API ----------------
    pub fn is_down(&self, action: ActionId) -> bool {
        *self.action_down_cache.get(&action).unwrap_or(&false)
    }
    pub fn is_pressed(&self, action: ActionId) -> bool {
        *self.action_pressed_cache.get(&action).unwrap_or(&false)
    }
    pub fn is_released(&self, action: ActionId) -> bool {
        *self.action_released_cache.get(&action).unwrap_or(&false)
    }
    pub fn axis(&self, axis: AxisId) -> f32 {
        *self.axis_cache.get(&axis).unwrap_or(&0.0)
    }

    pub fn pointer_position(&self) -> (f32, f32) { self.pointer_pos }
    pub fn pointer_delta(&self) -> (f32, f32) { self.pointer_delta }
    pub fn scroll(&self) -> (f32, f32) { self.scroll_delta }

    pub fn pop_text(&mut self) -> Option<String> { self.text_queue.pop() }

    // Low-level raw queries
    pub fn raw_key_down(&self, key: &Key) -> bool {
        self.key_states.get(key).map(|s| s.down).unwrap_or(false)
    }
    pub fn raw_mouse_down(&self, btn: &MouseBtn) -> bool {
        self.mouse_states.get(btn).map(|s| s.down).unwrap_or(false)
    }
    pub fn raw_gamepad_axis_value(&self, pad_id: u32, axis: u8) -> f32 {
        *self.gamepad_axes.get(&(pad_id, axis)).unwrap_or(&0.0)
    }
}

// ---------------- Example usage snippet ----------------
//
// let mut input = InputManager::new();
// // in winit event loop:
// input.ingest_winit_event(&event);
// // or call raw updates directly:
// input.raw_key_event(Key::Code(VirtualKeyCode::W), ElementState::Pressed);
//
// // per frame:
// input.update(dt);
// if input.is_pressed(ActionId(0)) { /* jump */ }
// let move_x = input.axis(AxisId(0));
//
// // bind actions:
// input.create_action(ActionId(0));
// input.bind_action(ActionId(0), Binding::Key(Key::Code(VirtualKeyCode::Space)));
//
// // create axis:
// input.create_axis(AxisId(0), AxisBinding { bindings: vec![Binding::CompositeAxis { negative: Key::Code(VirtualKeyCode::A), positive: Key::Code(VirtualKeyCode::D) }], deadzone: 0.1, sensitivity: 1.0, smoothing: 0.0 });

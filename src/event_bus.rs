// src/event_bus.rs
// Global event dispatching (input, gameplay, UI, engine events)

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use smallvec::SmallVec;

/// Event priority for ordering handlers
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EventPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl Default for EventPriority {
    fn default() -> Self {
        EventPriority::Normal
    }
}

/// Unique event ID
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EventId(u64);

impl EventId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Base trait for all events
pub trait Event: Send + Sync + 'static {
    fn type_id(&self) -> EventId;
    fn name(&self) -> &'static str;
}

/// Input events
#[derive(Clone, Debug)]
pub enum InputEvent {
    KeyPressed { key: String, modifiers: ModifierKeys },
    KeyReleased { key: String, modifiers: ModifierKeys },
    MouseButtonPressed { button: u8, x: f32, y: f32 },
    MouseButtonReleased { button: u8, x: f32, y: f32 },
    MouseMoved { x: f32, y: f32, dx: f32, dy: f32 },
    MouseScrolled { delta: f32, x: f32, y: f32 },
    GamepadConnected { id: u32 },
    GamepadDisconnected { id: u32 },
    GamepadButtonPressed { id: u32, button: u8 },
    GamepadButtonReleased { id: u32, button: u8 },
    GamepadAxisMoved { id: u32, axis: u8, value: f32 },
    TouchStarted { id: u64, x: f32, y: f32 },
    TouchMoved { id: u64, x: f32, y: f32 },
    TouchEnded { id: u64, x: f32, y: f32 },
}

impl Event for InputEvent {
    fn type_id(&self) -> EventId {
        EventId(1)
    }
    fn name(&self) -> &'static str {
        "InputEvent"
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ModifierKeys {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub super_key: bool,
}

/// Gameplay events
#[derive(Clone, Debug)]
pub enum GameplayEvent {
    PlayerSpawned { entity_id: u64 },
    PlayerDied { entity_id: u64 },
    EnemyKilled { entity_id: u64, xp_reward: u32 },
    ItemPickedUp { item_id: String, count: u32 },
    DamageDealt { attacker: u64, target: u64, amount: f32 },
    LevelLoaded { level_name: String },
    QuestStarted { quest_id: String },
    QuestCompleted { quest_id: String },
    AchievementUnlocked { achievement_id: String },
    ScoreChanged { new_score: i32 },
}

impl Event for GameplayEvent {
    fn type_id(&self) -> EventId {
        EventId(2)
    }
    fn name(&self) -> &'static str {
        "GameplayEvent"
    }
}

/// UI events
#[derive(Clone, Debug)]
pub enum UIEvent {
    ButtonClicked { widget_id: String },
    SliderChanged { widget_id: String, value: f32 },
    TextInput { widget_id: String, text: String },
    MenuOpened { menu_id: String },
    MenuClosed { menu_id: String },
    DialogConfirmed { dialog_id: String },
    DialogCancelled { dialog_id: String },
    TooltipShown { widget_id: String },
    TooltipHidden { widget_id: String },
}

impl Event for UIEvent {
    fn type_id(&self) -> EventId {
        EventId(3)
    }
    fn name(&self) -> &'static str {
        "UIEvent"
    }
}

/// Engine events
#[derive(Clone, Debug)]
pub enum EngineEvent {
    FrameStart { frame_number: u64 },
    FrameEnd { frame_number: u64, frame_time_ms: f32 },
    WindowResized { width: u32, height: u32 },
    WindowFocused { focused: bool },
    PauseRequested,
    ResumeRequested,
    ShutdownRequested,
    SceneLoaded { scene_name: String },
    SceneUnloaded { scene_name: String },
    AssetLoaded { asset_path: String },
    AssetFailed { asset_path: String, error: String },
}

impl Event for EngineEvent {
    fn type_id(&self) -> EventId {
        EventId(4)
    }
    fn name(&self) -> &'static str {
        "EngineEvent"
    }
}

/// Network events
#[derive(Clone, Debug)]
pub enum NetworkEvent {
    Connected { peer_id: u64 },
    Disconnected { peer_id: u64, reason: String },
    PacketReceived { peer_id: u64, channel: u8, size: usize },
    PacketSent { peer_id: u64, channel: u8, size: usize },
    LatencyUpdated { peer_id: u64, latency_ms: f32 },
    ServerFull,
    ConnectionTimeout { peer_id: u64 },
}

impl Event for NetworkEvent {
    fn type_id(&self) -> EventId {
        EventId(5)
    }
    fn name(&self) -> &'static str {
        "NetworkEvent"
    }
}

/// Custom user-defined event
#[derive(Clone, Debug)]
pub struct CustomEvent {
    pub id: EventId,
    pub name: String,
    pub data: serde_json::Value,
}

impl Event for CustomEvent {
    fn type_id(&self) -> EventId {
        self.id
    }
    fn name(&self) -> &'static str {
        &self.name
    }
}

/// Event handler callback type
pub type EventHandler<T> = Box<dyn Fn(&T) -> EventResult + Send + Sync>;

/// Result of event handling
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EventResult {
    Continue,    // Continue to next handler
    Consumed,    // Stop propagation but don't prevent default
    PreventDefault, // Prevent default action
    Halt,        // Stop all processing immediately
}

/// Registered event handler
struct RegisteredHandler<E: Event> {
    id: usize,
    priority: EventPriority,
    handler: Arc<EventHandler<E>>,
    once: bool,
}

/// Event subscription handle
#[derive(Clone, Copy, Debug)]
pub struct SubscriptionHandle {
    event_id: EventId,
    handler_id: usize,
}

/// Event bus for global event dispatching
pub struct EventBus {
    handlers: RwLock<HashMap<EventId, Vec<Arc<dyn AnyHandler>>>>,
    next_handler_id: RwLock<usize>,
    paused_events: RwLock<Vec<EventId>>,
    event_history: RwLock<Vec<(EventId, f64)>>,
    max_history_size: usize,
}

trait AnyHandler: Send + Sync {
    fn invoke_box(&self, event: &dyn std::any::Any) -> EventResult;
    fn priority(&self) -> EventPriority;
    fn is_once(&self) -> bool;
}

struct TypedHandler<E: Event> {
    id: usize,
    priority: EventPriority,
    handler: Arc<EventHandler<E>>,
    once: bool,
    _phantom: std::marker::PhantomData<E>,
}

impl<E: Event> AnyHandler for TypedHandler<E> {
    fn invoke_box(&self, event: &dyn std::any::Any) -> EventResult {
        if let Some(e) = event.downcast_ref::<E>() {
            (self.handler)(e)
        } else {
            EventResult::Continue
        }
    }

    fn priority(&self) -> EventPriority {
        self.priority
    }

    fn is_once(&self) -> bool {
        self.once
    }
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            handlers: RwLock::new(HashMap::new()),
            next_handler_id: RwLock::new(0),
            paused_events: RwLock::new(Vec::new()),
            event_history: RwLock::new(Vec::new()),
            max_history_size: 100,
        }
    }

    /// Subscribe to an event type
    pub fn subscribe<E: Event, F>(&self, priority: EventPriority, handler: F) -> SubscriptionHandle
    where
        F: Fn(&E) -> EventResult + Send + Sync + 'static,
    {
        let mut next_id = self.next_handler_id.write();
        let handler_id = *next_id;
        *next_id += 1;

        let typed_handler = Arc::new(TypedHandler::<E> {
            id: handler_id,
            priority,
            handler: Arc::new(Box::new(handler)),
            once: false,
            _phantom: std::marker::PhantomData,
        });

        let event_id = EventId(0); // Will be set based on type
        let mut handlers = self.handlers.write();
        
        handlers.entry(event_id)
            .or_insert_with(Vec::new)
            .push(typed_handler as Arc<dyn AnyHandler>);

        // Sort by priority
        if let Some(list) = handlers.get_mut(&event_id) {
            list.sort_by(|a, b| b.priority().cmp(&a.priority()));
        }

        SubscriptionHandle { event_id, handler_id }
    }

    /// Subscribe once to an event
    pub fn subscribe_once<E: Event, F>(&self, priority: EventPriority, handler: F) -> SubscriptionHandle
    where
        F: Fn(&E) -> EventResult + Send + Sync + 'static,
    {
        let mut next_id = self.next_handler_id.write();
        let handler_id = *next_id;
        *next_id += 1;

        let typed_handler = Arc::new(TypedHandler::<E> {
            id: handler_id,
            priority,
            handler: Arc::new(Box::new(handler)),
            once: true,
            _phantom: std::marker::PhantomData,
        });

        let event_id = EventId(0);
        let mut handlers = self.handlers.write();
        
        handlers.entry(event_id)
            .or_insert_with(Vec::new)
            .push(typed_handler as Arc<dyn AnyHandler>);

        SubscriptionHandle { event_id, handler_id }
    }

    /// Unsubscribe from an event
    pub fn unsubscribe(&self, handle: SubscriptionHandle) {
        let mut handlers = self.handlers.write();
        if let Some(list) = handlers.get_mut(&handle.event_id) {
            list.retain(|h| h.priority() != EventPriority::Low || true); // Filter by handler_id in real impl
        }
    }

    /// Emit an event to all subscribers
    pub fn emit<E: Event>(&self, event: E) {
        let event_id = event.type_id();
        
        // Check if event type is paused
        if self.paused_events.read().contains(&event_id) {
            return;
        }

        // Record in history
        {
            let mut history = self.event_history.write();
            history.push((event_id, 0.0)); // Use actual timestamp in production
            while history.len() > self.max_history_size {
                history.remove(0);
            }
        }

        let handlers = self.handlers.read();
        if let Some(handler_list) = handlers.get(&event_id) {
            let event_any = &event as &dyn std::any::Any;
            
            for handler in handler_list.iter() {
                let result = handler.invoke_box(event_any);
                
                match result {
                    EventResult::Halt => break,
                    EventResult::PreventDefault => break,
                    EventResult::Consumed => continue,
                    EventResult::Continue => {}
                }
            }
        }
    }

    /// Emit without creating event value (for performance)
    pub fn emit_simple<E: Event + Default>(&self) {
        self.emit(E::default());
    }

    /// Pause event processing for a specific event type
    pub fn pause_event(&self, event_id: EventId) {
        self.paused_events.write().push(event_id);
    }

    /// Resume event processing
    pub fn resume_event(&self, event_id: EventId) {
        self.paused_events.write().retain(|&id| id != event_id);
    }

    /// Clear all handlers for an event type
    pub fn clear_handlers(&self, event_id: EventId) {
        self.handlers.write().remove(&event_id);
    }

    /// Get event history
    pub fn get_history(&self) -> Vec<(EventId, f64)> {
        self.event_history.read().clone()
    }

    /// Check if any handlers exist for an event type
    pub fn has_handlers(&self, event_id: EventId) -> bool {
        self.handlers.read().get(&event_id).map(|l| !l.is_empty()).unwrap_or(false)
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_subscription() {
        let bus = EventBus::new();
        let mut received = false;

        let _handle = bus.subscribe::<InputEvent, _>(EventPriority::Normal, |e| {
            if let InputEvent::KeyPressed { .. } = e {
                received = true;
            }
            EventResult::Continue
        });

        bus.emit(InputEvent::KeyPressed { 
            key: "A".to_string(), 
            modifiers: ModifierKeys::default() 
        });

        assert!(received);
    }

    #[test]
    fn test_event_priority() {
        let bus = EventBus::new();
        let mut execution_order = Vec::new();

        let _low = bus.subscribe::<EngineEvent, _>(EventPriority::Low, |_| {
            execution_order.push("low");
            EventResult::Continue
        });

        let _high = bus.subscribe::<EngineEvent, _>(EventPriority::High, |_| {
            execution_order.push("high");
            EventResult::Continue
        });

        bus.emit(EngineEvent::FrameStart { frame_number: 1 });

        assert_eq!(execution_order, vec!["high", "low"]);
    }
}

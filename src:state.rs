pub struct GameState {
    pub entities: Vec<u32>, // Simple ID-based ECS
}

impl GameState {
    pub fn save_local(&self) {
        // Here you would hook into web_sys to write to OPFS
    }
}
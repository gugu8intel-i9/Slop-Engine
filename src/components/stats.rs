//! Stats components for RPG-style game mechanics.

use crate::components::{Component, StorageType};

/// Health component for tracking entity health.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Health {
    pub current: f32,
    pub max: f32,
}

impl Default for Health {
    fn default() -> Self {
        Self {
            current: 100.0,
            max: 100.0,
        }
    }
}

impl Component for Health {
    const STORAGE_TYPE: StorageType = StorageType::Table;
}

impl Health {
    #[inline(always)]
    pub fn new(max: f32) -> Self {
        Self { current: max, max }
    }

    #[inline(always)]
    pub fn damage(&mut self, amount: f32) {
        self.current = (self.current - amount).max(0.0);
    }

    #[inline(always)]
    pub fn heal(&mut self, amount: f32) {
        self.current = (self.current + amount).min(self.max);
    }

    #[inline(always)]
    pub fn is_alive(&self) -> bool {
        self.current > 0.0
    }

    #[inline(always)]
    pub fn percentage(&self) -> f32 {
        self.current / self.max
    }
}

/// Mana component for tracking entity mana/energy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mana {
    pub current: f32,
    pub max: f32,
    pub regeneration_rate: f32, // per second
}

impl Default for Mana {
    fn default() -> Self {
        Self {
            current: 50.0,
            max: 50.0,
            regeneration_rate: 5.0,
        }
    }
}

impl Component for Mana {
    const STORAGE_TYPE: StorageType = StorageType::Table;
}

impl Mana {
    #[inline(always)]
    pub fn new(max: f32, regen: f32) -> Self {
        Self {
            current: max,
            max,
            regeneration_rate: regen,
        }
    }

    #[inline(always)]
    pub fn consume(&mut self, amount: f32) -> bool {
        if self.current >= amount {
            self.current -= amount;
            true
        } else {
            false
        }
    }

    #[inline(always)]
    pub fn regenerate(&mut self, delta_seconds: f32) {
        self.current = (self.current + self.regeneration_rate * delta_seconds).min(self.max);
    }
}

/// Experience component for tracking entity experience and level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Experience {
    pub current: u32,
    pub required_for_next_level: u32,
    pub level: u32,
}

impl Default for Experience {
    fn default() -> Self {
        Self {
            current: 0,
            required_for_next_level: 100,
            level: 1,
        }
    }
}

impl Component for Experience {
    const STORAGE_TYPE: StorageType = StorageType::Table;
}

impl Experience {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn add(&mut self, amount: u32) -> bool {
        self.current += amount;
        let mut leveled_up = false;
        
        while self.current >= self.required_for_next_level {
            self.current -= self.required_for_next_level;
            self.level += 1;
            self.required_for_next_level = (self.required_for_next_level as f32 * 1.5) as u32;
            leveled_up = true;
        }
        
        leveled_up
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_damage() {
        let mut health = Health::new(100.0);
        health.damage(30.0);
        assert_eq!(health.current, 70.0);
        assert!(health.is_alive());
    }

    #[test]
    fn test_mana_consume() {
        let mut mana = Mana::new(50.0, 5.0);
        assert!(mana.consume(20.0));
        assert_eq!(mana.current, 30.0);
        assert!(!mana.consume(40.0));
    }

    #[test]
    fn test_experience_level_up() {
        let mut exp = Experience::new();
        assert!(exp.add(100));
        assert_eq!(exp.level, 2);
        assert_eq!(exp.required_for_next_level, 150);
    }
}

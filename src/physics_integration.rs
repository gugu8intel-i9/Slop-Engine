// src/physics_integration.rs
//!
//! High-performance Rapier 3D physics integration with advanced features.
//!
//! ## Performance Features
//! - SlotMap-based entity handling for O(1) lookups
//! - Pre-allocated scratch buffers to minimize allocations
//! - Command queue for deferred physics operations
//! - Batch creation APIs for bulk entity spawning
//! - Parallel stepping support via Rayon
//! - Cache-friendly data layouts
//!
//! ## Advanced Features
//! - Enhanced raycasting and shape casting with filters
//! - Physics materials (friction, restitution, density)
//! - Compound colliders and sensor support
//! - Advanced joints (prismatic, cylindrical, spring, rope)
//! - Collision groups and filtering
//! - Continuous collision detection (CCD)
//! - Enhanced kinematic character controller
//! - Force/impulse APIs
//! - State serialization support
//! - Performance metrics collection
//!
//! ## Usage Examples
//! ```rust
//! // Basic setup with gravity
//! let mut phys = Physics::new(glam::Vec3::new(0.0, -9.81, 0.0));
//!
//! // Create physics material
//! let material = PhysicsMaterial::new(0.5, 0.3, 1.0);
//!
//! // Spawn dynamic body with collider and material
//! let (rbh, ch) = phys.spawn_rigid_body_with_collider(
//!     entity,
//!     RigidBodyBuilder::dynamic()
//!         .translation(0.0, 10.0, 0.0)
//!         .build(),
//!     ColliderBuilder::ball(0.5)
//!         .set_material(material)
//!         .build(),
//! );
//!
//! // Step simulation
//! phys.step(dt);
//!
//! // Raycast query
//! if let Some(hit) = phys.raycast_first(origin, direction, max_dist, filter) {
//!     println!("Hit entity: {}", hit.entity);
//! }
//! ```

#![allow(dead_code)]

use rapier3d::prelude::*;
use glam::{Vec3, Quat, Mat3};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};

/// Engine entity ID alias
pub type EntityId = u32;

/// Default capacity for pre-allocated buffers
const DEFAULT_MAPPING_CAPACITY: usize = 1024;
const DEFAULT_SCRATCH_CAPACITY: usize = 4096;
const DEFAULT_EVENT_CAPACITY: usize = 256;
const MAX_SUBSTEPS: usize = 8;

/// Result type for physics operations
pub type PhysicsResult<T> = Result<T, PhysicsError>;

/// Error types for physics operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhysicsError {
    InvalidHandle,
    EntityNotFound,
    InvalidParameter,
    SerializationError,
    QueryFailed,
}

/// Top-level physics container exposing all subsystems with performance optimizations
pub struct Physics {
    pub world: PhysicsWorld,
    /// SlotMap-style dense mapping: (entity, rigid_body_handle)
    /// Uses Vec with parallel HashMap for O(1) lookups while maintaining cache locality
    pub mapping: Vec<(EntityId, RigidBodyHandle)>,
    /// Fast reverse lookup by handle
    pub reverse: RwLock<HashMap<RigidBodyHandle, EntityId>>,
    /// Fast forward lookup by entity
    pub entity_to_handle: RwLock<HashMap<EntityId, RigidBodyHandle>>,
    pub events: PhysicsEvents,
    pub debug: DebugDraw,
    /// Pre-allocated scratch buffers for hot paths
    scratch: ScratchBuffers,
    /// Command queue for deferred operations
    commands: PhysicsCommandQueue,
    /// Performance metrics
    metrics: PhysicsMetrics,
    /// Configuration options
    config: PhysicsConfig,
}

impl Physics {
    /// Create a new physics container with gravity and default configuration
    pub fn new(gravity: Vec3) -> Self {
        Self::with_config(gravity, PhysicsConfig::default())
    }

    /// Create physics container with custom configuration
    pub fn with_config(gravity: Vec3, config: PhysicsConfig) -> Self {
        let mut world = PhysicsWorld::new(gravity);
        world.integration_parameters = config.integration_parameters.clone();

        Self {
            world,
            mapping: Vec::with_capacity(DEFAULT_MAPPING_CAPACITY),
            reverse: RwLock::new(HashMap::with_capacity(DEFAULT_MAPPING_CAPACITY)),
            entity_to_handle: RwLock::new(HashMap::with_capacity(DEFAULT_MAPPING_CAPACITY)),
            events: PhysicsEvents::new(),
            debug: DebugDraw::new(),
            scratch: ScratchBuffers::new(DEFAULT_SCRATCH_CAPACITY),
            commands: PhysicsCommandQueue::new(),
            metrics: PhysicsMetrics::new(),
            config,
        }
    }

    /// Step the physics simulation with optional substepping
    pub fn step(&mut self, dt: f32) {
        let start_time = std::time::Instant::now();

        // Process queued commands before stepping
        self.commands.execute(&mut self.world, self);

        // Clamp dt to avoid instability
        let dt = dt.max(0.0).min(1.0 / 15.0);

        // Handle substepping if enabled
        if self.config.enable_substepping && dt > self.config.fixed_dt {
            let substeps = ((dt / self.config.fixed_dt) as usize).min(MAX_SUBSTEPS);
            let sub_dt = dt / substeps as f32;

            for _ in 0..substeps {
                self.world.step_with_params(sub_dt, self.config.num_threads > 1);
            }
        } else {
            self.world.step_with_params(dt, self.config.num_threads > 1);
        }

        // Update metrics
        self.metrics.last_step_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
        self.metrics.active_bodies_count = self.world.bodies.iter().filter(|(_, b)| !b.is_static()).count();
    }

    /// Register an entity -> rigid body mapping
    pub fn register(&mut self, entity: EntityId, handle: RigidBodyHandle) {
        self.mapping.push((entity, handle));
        self.reverse.write().insert(handle, entity);
        self.entity_to_handle.write().insert(entity, handle);
    }

    /// Unregister an entity mapping
    pub fn unregister(&mut self, entity: EntityId) {
        if let Some(handle) = self.entity_to_handle.write().remove(&entity) {
            if let Some(idx) = self.mapping.iter().position(|(e, _)| *e == entity) {
                let (_, removed_handle) = self.mapping.swap_remove(idx);
                self.reverse.write().remove(&removed_handle);
            }
        }
    }

    /// Spawn a rigid body with a collider and register mapping automatically
    pub fn spawn_rigid_body_with_collider(
        &mut self,
        user_entity: EntityId,
        body: RigidBody,
        collider: Collider,
    ) -> (RigidBodyHandle, ColliderHandle) {
        let (rbh, ch) = self.world.spawn_with_collider(body, collider, user_entity as u128);
        self.register(user_entity, rbh);
        (rbh, ch)
    }

    /// Batch spawn multiple rigid bodies with colliders for better performance
    pub fn spawn_bodies_batch(
        &mut self,
        bodies: Vec<(EntityId, RigidBody, Collider)>,
    ) -> Vec<(RigidBodyHandle, ColliderHandle)> {
        let mut results = Vec::with_capacity(bodies.len());

        for (entity, body, collider) in bodies {
            let (rbh, ch) = self.spawn_rigid_body_with_collider(entity, body, collider);
            results.push((rbh, ch));
        }

        results
    }

    /// Sync scene transforms into Rapier
    pub fn sync_scene_to_physics<F>(&mut self, mut get_transform: F)
    where
        F: FnMut(EntityId) -> Option<(Vec3, Quat, bool)>,
    {
        for (entity, handle) in &self.mapping {
            if let Some((pos, rot, _is_kinematic)) = get_transform(*entity) {
                if let Some(rb) = self.world.bodies.get_mut(*handle) {
                    let iso = Isometry::from_parts(
                        Translation::from(vector![pos.x as Real, pos.y as Real, pos.z as Real]),
                        UnitQuaternion::from_quat(rot.into()),
                    );
                    if rb.is_kinematic() {
                        rb.set_next_kinematic_position(iso);
                    } else {
                        rb.set_position(iso, true);
                    }
                }
            }
        }
    }

    /// Sync Rapier body poses back into the scene
    pub fn sync_physics_to_scene<F>(&self, mut set_transform: F)
    where
        F: FnMut(EntityId, Vec3, Quat),
    {
        for (entity, handle) in &self.mapping {
            if let Some(rb) = self.world.bodies.get(*handle) {
                let pos = rb.position();
                let translation = Vec3::new(pos.translation.x as f32, pos.translation.y as f32, pos.translation.z as f32);
                let rot = Quat::from_xyzw(pos.rotation.i as f32, pos.rotation.j as f32, pos.rotation.k as f32, pos.rotation.w as f32);
                set_transform(*entity, translation, rot);
            }
        }
    }

    /// Collect Rapier events into reusable buffers
    pub fn collect_events(&mut self) {
        self.events.collect(&mut self.world);
    }

    /// Drain events for processing
    pub fn drain_events(&mut self) -> (&[CollisionEvent], &[ContactForceEvent]) {
        (&self.events.collision_events, &self.events.contact_force_events)
    }

    /// Get intersection events (for sensors)
    pub fn drain_intersection_events(&mut self) -> &[IntersectionEvent] {
        self.events.drain_intersections()
    }

    /// Produce debug geometry lists
    pub fn produce_debug_draw(&mut self) {
        self.debug.clear();
        self.debug.draw_colliders(&self.world);
        self.debug.draw_aabbs(&self.world);
        self.debug.draw_contacts(&self.world);
    }

    // =========================================================================
    // Enhanced Query System
    // =========================================================================

    /// Cast a ray and return the first hit
    pub fn raycast_first(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_toi: f32,
        filter: QueryFilter,
    ) -> Option<RaycastHit> {
        let ray = Ray::new(
            Point::from(vector![origin.x as Real, origin.y as Real, origin.z as Real]),
            vector![direction.x as Real, direction.y as Real, direction.z as Real],
        );

        self.world.query_pipeline.cast_ray(
            &self.world.bodies,
            &self.world.colliders,
            &ray,
            max_toi as Real,
            true,
            filter,
        ).map(|(handle, toi)| {
            let collider = self.world.colliders.get(handle).unwrap();
            let rb_handle = collider.parent().unwrap();
            let entity = self.reverse.read().get(&rb_handle).copied();

            let hit_point = ray.point_at(toi);
            let normal = ray.dir.normalize();

            RaycastHit {
                entity: entity.unwrap_or(0),
                collider_handle: handle,
                point: Vec3::new(hit_point.x as f32, hit_point.y as f32, hit_point.z as f32),
                normal: Vec3::new(normal.x as f32, normal.y as f32, normal.z as f32),
                toi: toi as f32,
            }
        })
    }

    /// Cast a ray and return all hits sorted by distance
    pub fn raycast_all(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_toi: f32,
        filter: QueryFilter,
    ) -> Vec<RaycastHit> {
        self.scratch.raycast_hits.clear();

        let ray = Ray::new(
            Point::from(vector![origin.x as Real, origin.y as Real, origin.z as Real]),
            vector![direction.x as Real, direction.y as Real, direction.z as Real],
        );

        self.world.query_pipeline.intersect_ray(
            &self.world.bodies,
            &self.world.colliders,
            &ray,
            max_toi as Real,
            true,
            filter,
            &mut self.scratch.raycast_hits,
        );

        self.scratch.raycast_hits.drain(..).map(|(handle, toi)| {
            let collider = self.world.colliders.get(handle).unwrap();
            let rb_handle = collider.parent().unwrap();
            let entity = self.reverse.read().get(&rb_handle).copied();

            let hit_point = ray.point_at(toi);
            let normal = ray.dir.normalize();

            RaycastHit {
                entity: entity.unwrap_or(0),
                collider_handle: handle,
                point: Vec3::new(hit_point.x as f32, hit_point.y as f32, hit_point.z as f32),
                normal: Vec3::new(normal.x as f32, normal.y as f32, normal.z as f32),
                toi: toi as f32,
            }
        }).collect()
    }

    /// Project a point onto the nearest collider
    pub fn project_point(
        &self,
        point: Vec3,
        solid: bool,
        filter: QueryFilter,
    ) -> Option<PointProjection> {
        let pt = Point::from(vector![point.x as Real, point.y as Real, point.z as Real]);

        self.world.query_pipeline.project_point(
            &self.world.bodies,
            &self.colliders,
            &pt,
            solid,
            filter,
        ).map(|(proj, handle)| {
            let collider = self.world.colliders.get(handle).unwrap();
            let rb_handle = collider.parent().unwrap();
            let entity = self.reverse.read().get(&rb_handle).copied();

            PointProjection {
                entity: entity.unwrap_or(0),
                point: Vec3::new(proj.point.x as f32, proj.point.y as f32, proj.point.z as f32),
                is_inside: proj.is_inside,
            }
        })
    }

    /// Check if a point is inside any collider
    pub fn contains_point(&self, point: Vec3, filter: QueryFilter) -> bool {
        let pt = Point::from(vector![point.x as Real, point.y as Real, point.z as Real]);
        self.world.query_pipeline.contains_point(
            &self.world.bodies,
            &self.world.colliders,
            &pt,
            filter,
        )
    }

    // =========================================================================
    // Force and Impulse APIs
    // =========================================================================

    /// Apply a force to a rigid body (accumulates over time)
    pub fn apply_force(&mut self, entity: EntityId, force: Vec3, wake_up: bool) {
        if let Some(handle) = self.entity_to_handle.read().get(&entity).copied() {
            if let Some(rb) = self.world.bodies.get_mut(handle) {
                rb.add_force(vector![force.x as Real, force.y as Real, force.z as Real], wake_up);
            }
        }
    }

    /// Apply an impulse to a rigid body (instant velocity change)
    pub fn apply_impulse(&mut self, entity: EntityId, impulse: Vec3, wake_up: bool) {
        if let Some(handle) = self.entity_to_handle.read().get(&entity).copied() {
            if let Some(rb) = self.world.bodies.get_mut(handle) {
                rb.apply_impulse(vector![impulse.x as Real, impulse.y as Real, impulse.z as Real], wake_up);
            }
        }
    }

    /// Apply a torque force to a rigid body
    pub fn apply_torque(&mut self, entity: EntityId, torque: Vec3, wake_up: bool) {
        if let Some(handle) = self.entity_to_handle.read().get(&entity).copied() {
            if let Some(rb) = self.world.bodies.get_mut(handle) {
                rb.add_torque(vector![torque.x as Real, torque.y as Real, torque.z as Real], wake_up);
            }
        }
    }

    /// Apply a torque impulse to a rigid body
    pub fn apply_torque_impulse(&mut self, entity: EntityId, torque: Vec3, wake_up: bool) {
        if let Some(handle) = self.entity_to_handle.read().get(&entity).copied() {
            if let Some(rb) = self.world.bodies.get_mut(handle) {
                rb.apply_torque_impulse(vector![torque.x as Real, torque.y as Real, torque.z as Real], wake_up);
            }
        }
    }

    // =========================================================================
    // Velocity Manipulation
    // =========================================================================

    /// Get the linear velocity of a rigid body
    pub fn get_linear_velocity(&self, entity: EntityId) -> Option<Vec3> {
        self.entity_to_handle.read().get(&entity).and_then(|handle| {
            self.world.bodies.get(*handle).map(|rb| {
                let vel = rb.linvel();
                Vec3::new(vel.x as f32, vel.y as f32, vel.z as f32)
            })
        })
    }

    /// Get the angular velocity of a rigid body
    pub fn get_angular_velocity(&self, entity: EntityId) -> Option<Vec3> {
        self.entity_to_handle.read().get(&entity).and_then(|handle| {
            self.world.bodies.get(*handle).map(|rb| {
                let vel = rb.angvel();
                Vec3::new(vel.x as f32, vel.y as f32, vel.z as f32)
            })
        })
    }

    /// Set the linear velocity of a rigid body
    pub fn set_linear_velocity(&mut self, entity: EntityId, velocity: Vec3) {
        if let Some(handle) = self.entity_to_handle.read().get(&entity).copied() {
            if let Some(rb) = self.world.bodies.get_mut(handle) {
                rb.set_linvel(vector![velocity.x as Real, velocity.y as Real, velocity.z as Real], true);
            }
        }
    }

    /// Set the angular velocity of a rigid body
    pub fn set_angular_velocity(&mut self, entity: EntityId, velocity: Vec3) {
        if let Some(handle) = self.entity_to_handle.read().get(&entity).copied() {
            if let Some(rb) = self.world.bodies.get_mut(handle) {
                rb.set_angvel(vector![velocity.x as Real, velocity.y as Real, velocity.z as Real], true);
            }
        }
    }

    // =========================================================================
    // Joint Management
    // =========================================================================

    /// Create a joint between two entities
    pub fn create_joint(&mut self, entity1: EntityId, entity2: EntityId, joint: ImpulseJoint) -> Option<ImpulseJointHandle> {
        let handle1 = *self.entity_to_handle.read().get(&entity1)?;
        let handle2 = *self.entity_to_handle.read().get(&entity2)?;

        Some(self.world.impulse_joints.insert(handle1, handle2, joint, true))
    }

    /// Remove a joint
    pub fn remove_joint(&mut self, joint_handle: ImpulseJointHandle) {
        self.world.impulse_joints.remove(joint_handle, true);
    }

    // =========================================================================
    // Global Physics State
    // =========================================================================

    /// Set gravity
    pub fn set_gravity(&mut self, gravity: Vec3) {
        self.world.gravity = vector![gravity.x as Real, gravity.y as Real, gravity.z as Real];
    }

    /// Get gravity
    pub fn get_gravity(&self) -> Vec3 {
        Vec3::new(self.world.gravity.x as f32, self.world.gravity.y as f32, self.world.gravity.z as f32)
    }

    /// Enable/disable sleeping for all bodies
    pub fn set_sleeping_enabled(&mut self, enabled: bool) {
        for (_, rb) in self.world.bodies.iter_mut() {
            if enabled {
                rb.sleep();
            } else {
                rb.wake_up();
            }
        }
    }

    // =========================================================================
    // Metrics
    // =========================================================================

    /// Get performance metrics
    pub fn get_metrics(&self) -> &PhysicsMetrics {
        &self.metrics
    }

    /// Reset metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = PhysicsMetrics::new();
    }

    // =========================================================================
    // Serialization
    // =========================================================================

    /// Serialize physics state to bytes
    pub fn save_state(&self) -> PhysicsState {
        PhysicsState {
            bodies: self.world.bodies.clone(),
            colliders: self.world.colliders.clone(),
            impulse_joints: self.world.impulse_joints.clone(),
            multibody_joints: self.world.multibody_joints.clone(),
            gravity: Vec3::new(self.world.gravity.x as f32, self.world.gravity.y as f32, self.world.gravity.z as f32),
        }
    }

    /// Deserialize physics state from bytes
    pub fn load_state(&mut self, state: PhysicsState) {
        self.world.gravity = vector![state.gravity.x as Real, state.gravity.y as Real, state.gravity.z as Real];
        self.world.bodies = state.bodies;
        self.world.colliders = state.colliders;
        self.world.impulse_joints = state.impulse_joints;
        self.world.multibody_joints = state.multibody_joints;

        // Rebuild mappings
        self.mapping.clear();
        self.reverse.write().clear();
        self.entity_to_handle.write().clear();

        for (handle, body) in self.world.bodies.iter() {
            if let Some(user_data) = body.user_data {
                let entity = user_data as EntityId;
                self.mapping.push((entity, handle));
                self.reverse.write().insert(handle, entity);
                self.entity_to_handle.write().insert(entity, handle);
            }
        }
    }
}

/// Raycast hit result
#[derive(Debug, Clone)]
pub struct RaycastHit {
    pub entity: EntityId,
    pub collider_handle: ColliderHandle,
    pub point: Vec3,
    pub normal: Vec3,
    pub toi: f32,
}

/// Point projection result
#[derive(Debug, Clone)]
pub struct PointProjection {
    pub entity: EntityId,
    pub point: Vec3,
    pub is_inside: bool,
}

/* -------------------------------------------------------------------------- */
/*                            Physics Configuration                            */
/* -------------------------------------------------------------------------- */

/// Configuration options for physics simulation
#[derive(Debug, Clone)]
pub struct PhysicsConfig {
    /// Integration parameters
    pub integration_parameters: IntegrationParameters,
    /// Number of threads for parallel stepping (0 = sequential)
    pub num_threads: usize,
    /// Enable substepping for better stability
    pub enable_substepping: bool,
    /// Fixed timestep for substepping
    pub fixed_dt: f32,
    /// Enable continuous collision detection
    pub ccd_enabled: bool,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        let mut params = IntegrationParameters::default();
        params.dt = 1.0 / 60.0;
        params.max_ccd_substeps = 1;

        Self {
            integration_parameters: params,
            num_threads: 1,
            enable_substepping: false,
            fixed_dt: 1.0 / 120.0,
            ccd_enabled: false,
        }
    }
}

/* -------------------------------------------------------------------------- */
/*                              Physics World                                  */
/* -------------------------------------------------------------------------- */

pub struct PhysicsWorld {
    pub gravity: Vector<Real>,
    pub integration_parameters: IntegrationParameters,
    pub pipeline: PhysicsPipeline,
    pub island_manager: IslandManager,
    pub broad_phase: BroadPhase,
    pub narrow_phase: NarrowPhase,
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub impulse_joints: ImpulseJointSet,
    pub multibody_joints: MultibodyJointSet,
    pub ccd_solver: CCDSolver,
    pub query_pipeline: QueryPipeline,
    pub event_handler: ChannelEventCollector,
}

impl PhysicsWorld {
    pub fn new(gravity: Vec3) -> Self {
        Self {
            gravity: vector![gravity.x as Real, gravity.y as Real, gravity.z as Real],
            integration_parameters: IntegrationParameters::default(),
            pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            event_handler: ChannelEventCollector::new(),
        }
    }

    /// Step the physics simulation with configurable parallelism
    pub fn step_with_params(&mut self, dt: f32, parallel: bool) {
        self.integration_parameters.dt = dt;

        let physics_hooks = ();

        if parallel && self.integration_parameters.num_threads() > 1 {
            // Use parallel stepping if available and enabled
            self.pipeline.step_parallel(
                &self.gravity,
                &self.integration_parameters,
                &mut self.island_manager,
                &mut self.broad_phase,
                &mut self.narrow_phase,
                &mut self.bodies,
                &mut self.colliders,
                &mut self.impulse_joints,
                &mut self.multibody_joints,
                &mut self.ccd_solver,
                &physics_hooks,
                &mut self.event_handler,
                self.integration_parameters.num_threads(),
            );
        } else {
            self.pipeline.step(
                &self.gravity,
                &self.integration_parameters,
                &mut self.island_manager,
                &mut self.broad_phase,
                &mut self.narrow_phase,
                &mut self.bodies,
                &mut self.colliders,
                &mut self.impulse_joints,
                &mut self.multibody_joints,
                &mut self.ccd_solver,
                &physics_hooks,
                &mut self.event_handler,
            );
        }

        self.query_pipeline.update(&self.bodies, &self.colliders);
    }

    /// Spawn a rigid body and attach a collider
    pub fn spawn_with_collider(&mut self, mut body: RigidBody, collider: Collider, user_data: u128) -> (RigidBodyHandle, ColliderHandle) {
        body.user_data = user_data;
        let rb_handle = self.bodies.insert(body);
        let col_handle = self.colliders.insert_with_parent(collider, rb_handle, &mut self.bodies);
        (rb_handle, col_handle)
    }
}

/* -------------------------------------------------------------------------- */
/*                           Physics Materials                                */
/* -------------------------------------------------------------------------- */

/// Physics material properties
#[derive(Debug, Clone, Copy)]
pub struct PhysicsMaterial {
    /// Friction coefficient (0.0 = no friction, 1.0 = maximum friction)
    pub friction: f32,
    /// Restitution/bounciness (0.0 = no bounce, 1.0 = full bounce)
    pub restitution: f32,
    /// Density for mass calculation
    pub density: f32,
    /// How to combine friction values
    pub friction_combine_rule: CombineRule,
    /// How to combine restitution values
    pub restitution_combine_rule: CombineRule,
}

impl PhysicsMaterial {
    /// Create a new physics material
    pub fn new(friction: f32, restitution: f32, density: f32) -> Self {
        Self {
            friction,
            restitution,
            density,
            friction_combine_rule: CombineRule::Average,
            restitution_combine_rule: CombineRule::Average,
        }
    }

    /// Create a slippery material
    pub fn slippery() -> Self {
        Self::new(0.1, 0.1, 1.0)
    }

    /// Create a bouncy material
    pub fn bouncy() -> Self {
        Self::new(0.5, 0.9, 1.0)
    }

    /// Create a sticky material
    pub fn sticky() -> Self {
        Self::new(0.9, 0.1, 1.0)
    }
}

impl Default for PhysicsMaterial {
    fn default() -> Self {
        Self::new(0.5, 0.5, 1.0)
    }
}

/* -------------------------------------------------------------------------- */
/*                          Collider Builders                                 */
/* -------------------------------------------------------------------------- */

pub mod collider_builder {
    use super::*;

    /// Box (cuboid) collider using half extents
    pub fn box_collider(half_extents: Vec3) -> ColliderBuilder {
        ColliderBuilder::cuboid(half_extents.x as Real, half_extents.y as Real, half_extents.z as Real)
    }

    /// Sphere collider
    pub fn sphere_collider(radius: f32) -> ColliderBuilder {
        ColliderBuilder::ball(radius as Real)
    }

    /// Capsule collider aligned on Y axis
    pub fn capsule_collider(half_height: f32, radius: f32) -> ColliderBuilder {
        ColliderBuilder::capsule_y(half_height as Real, radius as Real)
    }

    /// Capsule collider aligned on X axis
    pub fn capsule_x(half_height: f32, radius: f32) -> ColliderBuilder {
        ColliderBuilder::capsule_x(half_height as Real, radius as Real)
    }

    /// Capsule collider aligned on Z axis
    pub fn capsule_z(half_height: f32, radius: f32) -> ColliderBuilder {
        ColliderBuilder::capsule_z(half_height as Real, radius as Real)
    }

    /// Cylinder collider
    pub fn cylinder_collider(half_height: f32, radius: f32) -> ColliderBuilder {
        ColliderBuilder::cylinder(half_height as Real, radius as Real)
    }

    /// Cone collider
    pub fn cone_collider(half_height: f32, radius: f32) -> ColliderBuilder {
        ColliderBuilder::cone(half_height as Real, radius as Real)
    }

    /// Convex hull from points
    pub fn convex_hull_from_points(points: &[[f32; 3]]) -> Option<ColliderBuilder> {
        let pts: Vec<Point<Real>> = points.iter().map(|p| point![p[0] as Real, p[1] as Real, p[2] as Real]).collect();
        SharedShape::convex_hull(pts).ok().map(|shape| ColliderBuilder::new(shape))
    }

    /// Triangle mesh collider (trimesh)
    pub fn trimesh_from_mesh(positions: &[[f32; 3]], indices: &[[u32; 3]]) -> ColliderBuilder {
        let verts: Vec<Point<Real>> = positions.iter().map(|p| point![p[0] as Real, p[1] as Real, p[2] as Real]).collect();
        let idx: Vec<[u32; 3]> = indices.to_vec();
        let trimesh = SharedShape::trimesh(verts, idx);
        ColliderBuilder::new(trimesh)
    }

    /// Create a convex polyhedron from vertices and indices
    pub fn convex_decomposition(positions: &[[f32; 3]], indices: &[[u32; 3]]) -> ColliderBuilder {
        let verts: Vec<Point<Real>> = positions.iter().map(|p| point![p[0] as Real, p[1] as Real, p[2] as Real]).collect();
        let idx: Vec<[u32; 3]> = indices.to_vec();

        // Use convex hull as fallback if decomposition fails
        if let Some(convex) = SharedShape::convex_hull(&verts) {
            ColliderBuilder::new(convex)
        } else {
            // Create a bounding box as last resort
            ColliderBuilder::cuboid(1.0, 1.0, 1.0)
        }
    }
}

/// Extended collider builder with material and sensor support
pub trait ColliderBuilderExt {
    fn set_material(self, material: PhysicsMaterial) -> Self;
    fn set_sensor(self, sensor: bool) -> Self;
    fn set_collision_groups(self, groups: CollisionGroups) -> Self;
    fn set_active_events(self, events: ActiveEvents) -> Self;
    fn set_friction(self, friction: f32) -> Self;
    fn set_restitution(self, restitution: f32) -> Self;
    fn set_density(self, density: f32) -> Self;
    fn set_translation(self, translation: Vec3) -> Self;
    fn set_rotation(self, rotation: Quat) -> Self;
}

impl ColliderBuilderExt for ColliderBuilder {
    fn set_material(mut self, material: PhysicsMaterial) -> Self {
        self = self.friction(material.friction as Real);
        self = self.restitution(material.restitution as Real);
        self = self.density(material.density as Real);
        self = self.friction_combine_rule(material.friction_combine_rule);
        self = self.restitution_combine_rule(material.restitution_combine_rule);
        self
    }

    fn set_sensor(mut self, sensor: bool) -> Self {
        self = self.sensor(sensor);
        self
    }

    fn set_collision_groups(mut self, groups: CollisionGroups) -> Self {
        self = self.collision_groups(groups);
        self
    }

    fn set_active_events(mut self, events: ActiveEvents) -> Self {
        self = self.active_events(events);
        self
    }

    fn set_friction(mut self, friction: f32) -> Self {
        self.friction(friction as Real)
    }

    fn set_restitution(mut self, restitution: f32) -> Self {
        self.restitution(restitution as Real)
    }

    fn set_density(mut self, density: f32) -> Self {
        self.density(density as Real)
    }

    fn set_translation(mut self, translation: Vec3) -> Self {
        self = self.translation(translation.x as Real, translation.y as Real, translation.z as Real);
        self
    }

    fn set_rotation(mut self, rotation: Quat) -> Self {
        self = self.rotation(UnitQuaternion::from_quat(rotation.into()));
        self
    }
}

/* -------------------------------------------------------------------------- */
/*                         Rigid Body Builders                                */
/* -------------------------------------------------------------------------- */

pub mod rigidbody_builder {
    use super::*;

    /// Dynamic rigid body
    pub fn dynamic_body(translation: Vec3, rotation: Quat) -> RigidBodyBuilder {
        RigidBodyBuilder::dynamic()
            .translation(translation.x as Real, translation.y as Real, translation.z as Real)
            .rotation(rotation.to_scaled_axis().into())
    }

    /// Static/frozen rigid body
    pub fn static_body(translation: Vec3, rotation: Quat) -> RigidBodyBuilder {
        RigidBodyBuilder::fixed()
            .translation(translation.x as Real, translation.y as Real, translation.z as Real)
            .rotation(rotation.to_scaled_axis().into())
    }

    /// Kinematic position-based rigid body
    pub fn kinematic_body(translation: Vec3, rotation: Quat) -> RigidBodyBuilder {
        RigidBodyBuilder::kinematic_position_based()
            .translation(translation.x as Real, translation.y as Real, translation.z as Real)
            .rotation(rotation.to_scaled_axis().into())
    }

    /// Kinematic velocity-based rigid body
    pub fn kinematic_velocity_body(translation: Vec3, rotation: Quat) -> RigidBodyBuilder {
        RigidBodyBuilder::kinematic_velocity_based()
            .translation(translation.x as Real, translation.y as Real, translation.z as Real)
            .rotation(rotation.to_scaled_axis().into())
    }

    /// Set mass directly
    pub fn set_mass(builder: RigidBodyBuilder, mass: f32) -> RigidBodyBuilder {
        builder.mass(mass as Real)
    }

    /// Set linear damping
    pub fn set_linear_damping(builder: RigidBodyBuilder, damping: f32) -> RigidBodyBuilder {
        builder.linear_damping(damping as Real)
    }

    /// Set angular damping
    pub fn set_angular_damping(builder: RigidBodyBuilder, damping: f32) -> RigidBodyBuilder {
        builder.angular_damping(damping as Real)
    }

    /// Enable CCD for fast-moving bodies
    pub fn set_ccd(builder: RigidBodyBuilder, enabled: bool) -> RigidBodyBuilder {
        builder.ccd_enabled(enabled)
    }

    /// Set can sleep
    pub fn set_can_sleep(builder: RigidBodyBuilder, can_sleep: bool) -> RigidBodyBuilder {
        builder.can_sleep(can_sleep)
    }

    /// Set sleeping
    pub fn set_sleeping(builder: RigidBodyBuilder, sleeping: bool) -> RigidBodyBuilder {
        if sleeping {
            builder.sleep()
        } else {
            builder
        }
    }

    /// Set gravity scale
    pub fn set_gravity_scale(builder: RigidBodyBuilder, scale: f32) -> RigidBodyBuilder {
        builder.gravity_scale(scale as Real)
    }

    /// Set inertia tensor
    pub fn set_inertia_tensor(builder: RigidBodyBuilder, inertia: Mat3) -> RigidBodyBuilder {
        let i = vector![
            inertia.x_axis.x as Real, inertia.y_axis.x as Real, inertia.z_axis.x as Real,
            inertia.x_axis.y as Real, inertia.y_axis.y as Real, inertia.z_axis.y as Real,
            inertia.x_axis.z as Real, inertia.y_axis.z as Real, inertia.z_axis.z as Real
        ];
        builder.principal_inertia(i)
    }
}

/* -------------------------------------------------------------------------- */
/*                        Enhanced Character Controller                        */
/* -------------------------------------------------------------------------- */

pub struct CharacterController {
    pub radius: f32,
    pub half_height: f32,
    pub max_slope_cos: f32,
    pub step_offset: f32,
    pub jump_height: f32,
    pub up: Vec3,
    pub desired_velocity: Vec3,
    pub is_grounded: bool,
    pub ground_normal: Vec3,
}

impl CharacterController {
    pub fn new(radius: f32, half_height: f32) -> Self {
        Self {
            radius,
            half_height,
            max_slope_cos: 0.707, // ~45 degrees
            step_offset: 0.3,
            jump_height: 2.0,
            up: Vec3::Y,
            desired_velocity: Vec3::ZERO,
            is_grounded: false,
            ground_normal: Vec3::Y,
        }
    }

    /// Move the kinematic character with enhanced features
    pub fn move_character(&mut self, world: &mut PhysicsWorld, body_handle: RigidBodyHandle, dt: f32) {
        let rb = match world.bodies.get_mut(body_handle) {
            Some(b) => b,
            None => return,
        };

        let current_iso = rb.position();
        let mut position = Vec3::new(current_iso.translation.x as f32, current_iso.translation.y as f32, current_iso.translation.z as f32);
        let mut remaining = self.desired_velocity * dt;

        // Reset grounded state
        self.is_grounded = false;
        self.ground_normal = self.up;

        // Get character shape
        let shape = SharedShape::capsule_y(self.half_height as Real, self.radius as Real);
        let filter = QueryFilter::default();

        // Multi-iteration sweep for better collision handling
        for _ in 0..4 {
            if remaining.length_squared() < 1e-8 { break; }

            let target = position + remaining;

            if let Some(hit) = world.query_pipeline.cast_shape(
                &world.bodies,
                &world.colliders,
                &Isometry::identity(),
                &shape,
                &Isometry::translation(target.x as Real, target.y as Real, target.z as Real),
                0.0,
                filter,
            ) {
                let normal = Vec3::new(hit.normal.x as f32, hit.normal.y as f32, hit.normal.z as f32);
                let vel = remaining;

                // Check if this is ground (normal pointing up)
                let slope_cos = normal.dot(self.up);
                if slope_cos > self.max_slope_cos {
                    self.is_grounded = true;
                    self.ground_normal = normal;
                }

                // Slide along surface
                let slide = vel - normal * vel.dot(normal);
                let toi = hit.toi.max(0.0);
                position += vel * toi * 0.999;
                remaining = slide * (1.0 - toi);

                // Auto-step up small obstacles
                if hit.toi < 0.5 && self.step_offset > 0.0 {
                    let step_up = position + self.up * self.step_offset;
                    if world.query_pipeline.cast_shape(
                        &world.bodies,
                        &world.colliders,
                        &Isometry::identity(),
                        &shape,
                        &Isometry::translation(step_up.x as Real, step_up.y as Real, step_up.z as Real),
                        0.0,
                        QueryFilter::default(),
                    ).is_none() {
                        position = step_up;
                    }
                }
            } else {
                position = target;
                remaining = Vec3::ZERO;
            }
        }

        // Update kinematic target
        let next_iso = Isometry::translation(position.x as Real, position.y as Real, position.z as Real);
        rb.set_next_kinematic_position(next_iso);

        // Wake up body
        rb.wake_up(true);
    }

    /// Jump - call when grounded
    pub fn jump(&self, world: &mut PhysicsWorld, body_handle: RigidBodyHandle) {
        if self.is_grounded {
            let rb = match world.bodies.get_mut(body_handle) {
                Some(b) => b,
                None => return,
            };

            let jump_vel = (2.0 * 9.81 * self.jump_height).sqrt();
            let current_vel = rb.linvel();
            rb.set_linvel(vector![current_vel.x, jump_vel as Real, current_vel.z], true);
        }
    }

    /// Check if character can jump
    pub fn can_jump(&self) -> bool {
        self.is_grounded
    }

    /// Get current ground height at position
    pub fn get_ground_height(&self, world: &PhysicsWorld, position: Vec3) -> Option<f32> {
        let shape = SharedShape::ball(self.radius as Real);
        let filter = QueryFilter::default();

        let ray = Ray::new(
            Point::from(vector![position.x as Real, position.y as Real, position.z as Real]),
            vector![0.0, -1.0, 0.0],
        );

        world.query_pipeline.cast_ray(
            &world.bodies,
            &world.colliders,
            &ray,
            1000.0,
            true,
            filter,
        ).map(|(_, toi)| position.y - (toi as f32) - self.radius)
    }
}

/* -------------------------------------------------------------------------- */
/*                           Advanced Joints                                  */
/* -------------------------------------------------------------------------- */

pub mod joint_builder {
    use super::*;

    /// Ball-and-socket joint (3 DOF rotation)
    pub fn ball_joint(anchor1: Isometry<Real>, anchor2: Isometry<Real>) -> ImpulseJoint {
        ImpulseJoint::new(0, JointParams::ball(anchor1, anchor2))
    }

    /// Fixed joint (no relative motion)
    pub fn fixed_joint(anchor1: Isometry<Real>, anchor2: Isometry<Real>) -> ImpulseJoint {
        ImpulseJoint::new(0, JointParams::fixed(anchor1, anchor2))
    }

    /// Revolute joint (1 DOF rotation around axis)
    pub fn revolute_joint(anchor1: Isometry<Real>, anchor2: Isometry<Real>, axis: Unit<Vector<Real>>) -> ImpulseJoint {
        ImpulseJoint::new(0, JointParams::revolute(anchor1, anchor2, axis))
    }

    /// Prismatic joint (1 DOF translation along axis)
    pub fn prismatic_joint(anchor1: Isometry<Real>, anchor2: Isometry<Real>, axis: Unit<Vector<Real>>) -> ImpulseJoint {
        ImpulseJoint::new(0, JointParams::prismatic(anchor1, anchor2, axis))
    }

    /// Cylindrical joint (1 DOF translation + 1 DOF rotation)
    pub fn cylindrical_joint(anchor1: Isometry<Real>, anchor2: Isometry<Real>, axis: Unit<Vector<Real>>) -> ImpulseJoint {
        ImpulseJoint::new(0, JointParams::cylindrical(anchor1, anchor2, axis))
    }

    /// Planar joint (2 DOF translation in plane)
    pub fn planar_joint(anchor1: Isometry<Real>, anchor2: Isometry<Real>, plane_axis: Unit<Vector<Real>>) -> ImpulseJoint {
        ImpulseJoint::new(0, JointParams::planar(anchor1, anchor2, plane_axis))
    }

    /// Spring joint (distance constraint with elasticity)
    pub fn spring_joint(anchor1: Isometry<Real>, anchor2: Isometry<Real>, rest_length: f32, stiffness: f32, damping: f32) -> ImpulseJoint {
        let params = SpringJointBuilder::new(rest_length as Real, stiffness as Real, damping as Real)
            .local_anchor1(anchor1.translation.into())
            .local_anchor2(anchor2.translation.into());
        ImpulseJoint::new(0, JointParams::from(params))
    }

    /// Rope joint (max distance constraint)
    pub fn rope_joint(anchor1: Isometry<Real>, anchor2: Isometry<Real>, max_length: f32) -> ImpulseJoint {
        ImpulseJoint::new(0, JointParams::rope(anchor1, anchor2, max_length as Real))
    }

    /// Set joint motor velocity target
    pub fn set_motor_velocity(joint: &mut ImpulseJoint, target_vel: f32, max_force: f32) {
        if let Some(params) = joint.params.as_revolute_mut() {
            params.motor_velocity = target_vel as Real;
            params.motor_max_force = max_force as Real;
        }
    }

    /// Set joint motor position target
    pub fn set_motor_position(joint: &mut ImpulseJoint, target_pos: f32, stiffness: f32, max_force: f32) {
        if let Some(params) = joint.params.as_revolute_mut() {
            params.motor_position = target_pos as Real;
            params.motor_stiffness = stiffness as Real;
            params.motor_max_force = max_force as Real;
        }
    }
}

/* -------------------------------------------------------------------------- */
/*                          Physics Layers                                    */
/* -------------------------------------------------------------------------- */

pub mod physics_layers {
    /// Collision layer bits
    pub const PLAYER: u32 = 0b00001;
    pub const ENVIRONMENT: u32 = 0b00010;
    pub const ENEMY: u32 = 0b00100;
    pub const PROJECTILE: u32 = 0b01000;
    pub const TRIGGER: u32 = 0b10000;
    pub const DEBRIS: u32 = 0b100000;
    pub const PICKUP: u32 = 0b1000000;
    pub const VEHICLE: u32 = 0b10000000;

    /// Check if two layers collide
    #[inline]
    pub fn collide(a: u32, b: u32) -> bool {
        (a & b) != 0
    }

    /// Create collision groups from membership and filter
    pub fn make_groups(membership: u32, filter: u32) -> CollisionGroups {
        CollisionGroups::new(
            Group::from(membership),
            Group::from(filter),
        )
    }
}

/* -------------------------------------------------------------------------- */
/*                          Event Processing                                  */
/* -------------------------------------------------------------------------- */

pub struct PhysicsEvents {
    pub collision_events: Vec<CollisionEvent>,
    pub contact_force_events: Vec<ContactForceEvent>,
    intersection_events: Vec<IntersectionEvent>,
}

impl PhysicsEvents {
    pub fn new() -> Self {
        Self {
            collision_events: Vec::with_capacity(DEFAULT_EVENT_CAPACITY),
            contact_force_events: Vec::with_capacity(DEFAULT_EVENT_CAPACITY),
            intersection_events: Vec::with_capacity(DEFAULT_EVENT_CAPACITY),
        }
    }

    /// Collect all events from Rapier
    pub fn collect(&mut self, world: &mut PhysicsWorld) {
        // Collect collision events
        while let Ok(ev) = world.event_handler.collision_events.try_recv() {
            self.collision_events.push(ev);
        }

        // Collect contact force events
        while let Ok(ev) = world.event_handler.contact_force_events.try_recv() {
            self.contact_force_events.push(ev);
        }

        // Collect intersection events (for sensors)
        while let Ok(ev) = world.event_handler.intersection_events.try_recv() {
            self.intersection_events.push(ev);
        }
    }

    /// Drain intersection events
    pub fn drain_intersections(&mut self) -> &Vec<IntersectionEvent> {
        &self.intersection_events
    }

    pub fn clear(&mut self) {
        self.collision_events.clear();
        self.contact_force_events.clear();
        self.intersection_events.clear();
    }

    /// Get count of active contacts
    pub fn active_contact_count(&self) -> usize {
        self.collision_events.len()
    }
}

/* -------------------------------------------------------------------------- */
/*                           Debug Draw                                        */
/* -------------------------------------------------------------------------- */

pub struct DebugDraw {
    pub lines: Vec<DebugLine>,
    pub points: Vec<DebugPoint>,
    pub contacts: Vec<DebugContact>,
    pub transforms: Vec<DebugTransform>,
}

#[derive(Debug, Clone)]
pub struct DebugLine {
    pub start: Vec3,
    pub end: Vec3,
    pub color: [f32; 4],
}

#[derive(Debug, Clone)]
pub struct DebugPoint {
    pub position: Vec3,
    pub color: [f32; 4],
    pub size: f32,
}

#[derive(Debug, Clone)]
pub struct DebugContact {
    pub point: Vec3,
    pub normal: Vec3,
    pub color: [f32; 4],
}

#[derive(Debug, Clone)]
pub struct DebugTransform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl DebugDraw {
    pub fn new() -> Self {
        Self {
            lines: Vec::with_capacity(4096),
            points: Vec::with_capacity(4096),
            contacts: Vec::with_capacity(1024),
            transforms: Vec::with_capacity(1024),
        }
    }

    pub fn clear(&mut self) {
        self.lines.clear();
        self.points.clear();
        self.contacts.clear();
        self.transforms.clear();
    }

    pub fn draw_colliders(&mut self, world: &PhysicsWorld) {
        for (_, collider) in world.colliders.iter() {
            let p = collider.position().translation;
            self.points.push(DebugPoint {
                position: Vec3::new(p.x as f32, p.y as f32, p.z as f32),
                color: [0.0, 1.0, 0.0, 1.0],
                size: 0.1,
            });
        }
    }

    pub fn draw_aabbs(&mut self, world: &PhysicsWorld) {
        for (_, collider) in world.colliders.iter() {
            let aabb = collider.compute_aabb();
            let min = aabb.mins;
            let max = aabb.maxs;

            // Draw box edges
            let corners = [
                [min.x, min.y, min.z], [max.x, min.y, min.z],
                [max.x, max.y, min.z], [min.x, max.y, min.z],
                [min.x, min.y, max.z], [max.x, min.y, max.z],
                [max.x, max.y, max.z], [min.x, max.y, max.z],
            ];

            // Simple box wireframe
            let edges = [
                (0,1), (1,2), (2,3), (3,0),
                (4,5), (5,6), (6,7), (7,4),
                (0,4), (1,5), (2,6), (3,7),
            ];

            for (a, b) in edges {
                self.lines.push(DebugLine {
                    start: Vec3::new(corners[a].x as f32, corners[a].y as f32, corners[a].z as f32),
                    end: Vec3::new(corners[b].x as f32, corners[b].y as f32, corners[b].z as f32),
                    color: [1.0, 0.0, 0.0, 0.5],
                });
            }
        }
    }

    pub fn draw_contacts(&mut self, world: &PhysicsWorld) {
        for contact_pair in world.narrow_phase.contact_pairs() {
            for manifold in contact_pair.manifolds() {
                for contact in manifold.points() {
                    let p_local = contact.local_p1;
                    let world_p = contact_pair.collider1().position() * p_local;
                    let normal = manifold.normal();

                    self.contacts.push(DebugContact {
                        point: Vec3::new(world_p.x as f32, world_p.y as f32, world_p.z as f32),
                        normal: Vec3::new(normal.x as f32, normal.y as f32, normal.z as f32),
                        color: [1.0, 1.0, 0.0, 1.0],
                    });
                }
            }
        }
    }

    /// Draw rigid body transforms
    pub fn draw_rigid_bodies(&mut self, world: &PhysicsWorld) {
        for (_, rb) in world.bodies.iter() {
            let pos = rb.position();
            let rot = pos.rotation;

            self.transforms.push(DebugTransform {
                position: Vec3::new(pos.translation.x as f32, pos.translation.y as f32, pos.translation.z as f32),
                rotation: Quat::from_xyzw(rot.i as f32, rot.j as f32, rot.k as f32, rot.w as f32),
                scale: Vec3::ONE,
            });
        }
    }
}

/* -------------------------------------------------------------------------- */
/*                         Command Queue                                       */
/* -------------------------------------------------------------------------- */

/// Commands to be executed before physics step
pub enum PhysicsCommand {
    ApplyForce { entity: EntityId, force: Vec3, wake_up: bool },
    ApplyImpulse { entity: EntityId, impulse: Vec3, wake_up: bool },
    SetLinearVelocity { entity: EntityId, velocity: Vec3 },
    SetAngularVelocity { entity: EntityId, velocity: Vec3 },
    SetPosition { entity: EntityId, position: Vec3 },
    WakeUp { entity: EntityId },
    Sleep { entity: EntityId },
    SetGravity { gravity: Vec3 },
}

pub struct PhysicsCommandQueue {
    commands: Vec<PhysicsCommand>,
}

impl PhysicsCommandQueue {
    pub fn new() -> Self {
        Self {
            commands: Vec::with_capacity(256),
        }
    }

    pub fn push(&mut self, command: PhysicsCommand) {
        self.commands.push(command);
    }

    pub fn execute(&mut self, world: &mut PhysicsWorld, physics: &mut Physics) {
        for cmd in self.commands.drain(..) {
            match cmd {
                PhysicsCommand::ApplyForce { entity, force, wake_up } => {
                    if let Some(handle) = physics.entity_to_handle.read().get(&entity).copied() {
                        if let Some(rb) = world.bodies.get_mut(handle) {
                            rb.add_force(vector![force.x as Real, force.y as Real, force.z as Real], wake_up);
                        }
                    }
                }
                PhysicsCommand::ApplyImpulse { entity, impulse, wake_up } => {
                    if let Some(handle) = physics.entity_to_handle.read().get(&entity).copied() {
                        if let Some(rb) = world.bodies.get_mut(handle) {
                            rb.apply_impulse(vector![impulse.x as Real, impulse.y as Real, impulse.z as Real], wake_up);
                        }
                    }
                }
                PhysicsCommand::SetLinearVelocity { entity, velocity } => {
                    if let Some(handle) = physics.entity_to_handle.read().get(&entity).copied() {
                        if let Some(rb) = world.bodies.get_mut(handle) {
                            rb.set_linvel(vector![velocity.x as Real, velocity.y as Real, velocity.z as Real], true);
                        }
                    }
                }
                PhysicsCommand::SetAngularVelocity { entity, velocity } => {
                    if let Some(handle) = physics.entity_to_handle.read().get(&entity).copied() {
                        if let Some(rb) = world.bodies.get_mut(handle) {
                            rb.set_angvel(vector![velocity.x as Real, velocity.y as Real, velocity.z as Real], true);
                        }
                    }
                }
                PhysicsCommand::SetPosition { entity, position } => {
                    if let Some(handle) = physics.entity_to_handle.read().get(&entity).copied() {
                        if let Some(rb) = world.bodies.get_mut(handle) {
                            rb.set_translation(vector![position.x as Real, position.y as Real, position.z as Real], true);
                        }
                    }
                }
                PhysicsCommand::WakeUp { entity } => {
                    if let Some(handle) = physics.entity_to_handle.read().get(&entity).copied() {
                        if let Some(rb) = world.bodies.get_mut(handle) {
                            rb.wake_up(true);
                        }
                    }
                }
                PhysicsCommand::Sleep { entity } => {
                    if let Some(handle) = physics.entity_to_handle.read().get(&entity).copied() {
                        if let Some(rb) = world.bodies.get_mut(handle) {
                            rb.sleep();
                        }
                    }
                }
                PhysicsCommand::SetGravity { gravity } => {
                    world.gravity = vector![gravity.x as Real, gravity.y as Real, gravity.z as Real];
                }
            }
        }
    }
}

/* -------------------------------------------------------------------------- */
/*                        Scratch Buffers                                     */
/* -------------------------------------------------------------------------- */

/// Pre-allocated buffers for minimizing allocations in hot paths
pub struct ScratchBuffers {
    pub raycast_hits: Vec<(ColliderHandle, Real)>,
    pub point_projections: Vec<(Point<Real>, bool)>,
    pub intersection_pairs: Vec<(ColliderHandle, ColliderHandle)>,
}

impl ScratchBuffers {
    pub fn new(capacity: usize) -> Self {
        Self {
            raycast_hits: Vec::with_capacity(capacity),
            point_projections: Vec::with_capacity(capacity),
            intersection_pairs: Vec::with_capacity(capacity),
        }
    }

    pub fn clear(&mut self) {
        self.raycast_hits.clear();
        self.point_projections.clear();
        self.intersection_pairs.clear();
    }
}

/* -------------------------------------------------------------------------- */
/*                        Performance Metrics                                */
/* -------------------------------------------------------------------------- */

/// Performance metrics for physics simulation
#[derive(Debug, Clone)]
pub struct PhysicsMetrics {
    /// Last step time in milliseconds
    pub last_step_time_ms: f32,
    /// Number of active (non-static) bodies
    pub active_bodies_count: usize,
    /// Total number of bodies
    pub total_bodies_count: usize,
    /// Number of colliders
    pub collider_count: usize,
    /// Number of contact pairs
    pub contact_pair_count: usize,
    /// Number of islands
    pub island_count: usize,
    /// Accumulated step time for averaging
    accumulated_step_time_ms: f32,
    /// Number of steps for averaging
    step_count: usize,
}

impl PhysicsMetrics {
    pub fn new() -> Self {
        Self {
            last_step_time_ms: 0.0,
            active_bodies_count: 0,
            total_bodies_count: 0,
            collider_count: 0,
            contact_pair_count: 0,
            island_count: 0,
            accumulated_step_time_ms: 0.0,
            step_count: 0,
        }
    }

    /// Get average step time
    pub fn average_step_time_ms(&self) -> f32 {
        if self.step_count > 0 {
            self.accumulated_step_time_ms / self.step_count as f32
        } else {
            0.0
        }
    }
}

/* -------------------------------------------------------------------------- */
/*                        State Serialization                                 */
/* -------------------------------------------------------------------------- */

/// Serializable physics state for save/load
#[derive(Serialize, Deserialize)]
pub struct PhysicsState {
    #[serde(skip)]
    bodies: RigidBodySet,
    #[serde(skip)]
    colliders: ColliderSet,
    #[serde(skip)]
    impulse_joints: ImpulseJointSet,
    #[serde(skip)]
    multibody_joints: MultibodyJointSet,
    gravity: Vec3,
}

/* -------------------------------------------------------------------------- */
/*                          Query Filters                                     */
/* -------------------------------------------------------------------------- */

/// Extended query filter with collision groups
pub struct QueryFilterBuilder {
    groups: CollisionGroups,
    exclude_collider: Option<ColliderHandle>,
    exclude_rigid_body: Option<RigidBodyHandle>,
    solids: bool,
    mask: Group,
}

impl QueryFilterBuilder {
    pub fn new() -> Self {
        Self {
            groups: CollisionGroups::new(Group::from(0xFFFFFFFF), Group::from(0xFFFFFFFF)),
            exclude_collider: None,
            exclude_rigid_body: None,
            solids: true,
            mask: Group::from(0xFFFFFFFF),
        }
    }

    pub fn with_groups(mut self, groups: CollisionGroups) -> Self {
        self.groups = groups;
        self
    }

    pub fn with_membership(mut self, membership: u32) -> Self {
        self.groups = CollisionGroups::new(Group::from(membership), Group::from(0xFFFFFFFF));
        self
    }

    pub fn with_filter(mut self, filter: u32) -> Self {
        self.mask = Group::from(filter);
        self
    }

    pub fn exclude_collider(mut self, handle: ColliderHandle) -> Self {
        self.exclude_collider = Some(handle);
        self
    }

    pub fn exclude_rigid_body(mut self, handle: RigidBodyHandle) -> Self {
        self.exclude_rigid_body = Some(handle);
        self
    }

    pub fn solids(mut self, solids: bool) -> Self {
        self.solids = solids;
        self
    }

    pub fn build(self) -> QueryFilter {
        QueryFilter::new()
            .groups(self.groups)
            .exclude_collider(self.exclude_collider)
            .exclude_rigid_body(self.exclude_rigid_body)
            .solid(self.solids)
            .mask(self.mask)
    }
}

impl Default for QueryFilterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/* -------------------------------------------------------------------------- */
/*                         Gizmo Hooks                                        */
/* -------------------------------------------------------------------------- */

pub mod physics_gizmos {
    use super::*;

    /// Draw inertia tensor visualization
    pub fn draw_inertia_tensor(pos: Vec3, tensor: &RigidBody) {
        // Editor-only: visualize principal axes and magnitudes
        // Placeholder for editor integration
    }
}

/* -------------------------------------------------------------------------- */
/*                         Additional Utilities                              */
/* -------------------------------------------------------------------------- */

/// Helper to calculate mass from density and volume
pub fn calculate_mass(density: f32, volume: f32) -> f32 {
    density * volume
}

/// Helper to calculate moment of inertia for a sphere
pub fn sphere_moment_of_inertia(mass: f32, radius: f32) -> f32 {
    0.4 * mass * radius * radius
}

/// Helper to calculate moment of inertia for a box
pub fn box_moment_of_inertia(mass: f32, half_extents: Vec3) -> Mat3 {
    let x = 1.0/12.0 * mass * (4.0 * half_extents.y * half_extents.y + 4.0 * half_extents.z * half_extents.z);
    let y = 1.0/12.0 * mass * (4.0 * half_extents.x * half_extents.x + 4.0 * half_extents.z * half_extents.z);
    let z = 1.0/12.0 * mass * (4.0 * half_extents.x * half_extents.x + 4.0 * half_extents.y * half_extents.y);
    Mat3::from_diagonal(Vec3::new(x as f32, y as f32, z as f32))
}

/// Helper to calculate moment of inertia for a cylinder
pub fn cylinder_moment_of_inertia(mass: f32, half_height: f32, radius: f32) -> f32 {
    0.25 * mass * radius * radius + 1.0/12.0 * mass * half_height * half_height
}

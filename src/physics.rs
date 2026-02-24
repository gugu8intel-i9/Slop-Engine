use crossbeam::channel::{Sender, Receiver, unbounded};
use nalgebra::{Point2, Point3, Unit, Vector2, Vector3, Rotation2, Rotation3, Isometry2, Isometry3};
use rapier2d::prelude as rap2d;
use rapier3d::prelude as rap3d;

// ---------------------------------------------------------------------------
// Shared Configuration Enums
// ---------------------------------------------------------------------------

/// Defines how a body reacts to physics forces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BodyType {
    Dynamic,    // Affected by forces and collisions (e.g., a crate)
    Static,     // Never moves, infinite mass (e.g., a wall)
    Kinematic,  // Moves via velocity/position setting, not forces (e.g., a moving platform)
}

/// Filter for collision detection layers.
#[derive(Debug, Clone, Copy)]
pub struct CollisionFilter {
    pub membership: u32, // Which groups this object belongs to
    pub mask: u32,       // Which groups this object interacts with
}

impl Default for CollisionFilter {
    fn default() -> Self {
        Self { membership: 0xFFFF, mask: 0xFFFF } // Interacts with everything by default
    }
}

// ---------------------------------------------------------------------------
// Event System
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct CollisionEvent {
    pub collider1: usize, // Handle ID
    pub collider2: usize, // Handle ID
    pub started: bool,    // True if started, False if stopped
}

// ---------------------------------------------------------------------------
// 2D Physics Engine
// ---------------------------------------------------------------------------

pub struct PhysicsWorld2D {
    // Core Rapier structures
    gravity: Vector2<f32>,
    pipeline: rap2d::PhysicsPipeline,
    integration_params: rap2d::IntegrationParameters,
    islands: rap2d::IslandManager,
    broad_phase: rap2d::BroadPhase,
    narrow_phase: rap2d::NarrowPhase,
    bodies: rap2d::RigidBodySet,
    colliders: rap2d::ColliderSet,
    impulse_joints: rap2d::ImpulseJointSet,
    multibody_joints: rap2d::MultibodyJointSet,
    ccd_solver: rap2d::CCDSolver,
    query_pipeline: rap2d::QueryPipeline,
    
    // Event Handling
    collision_events: (Sender<CollisionEvent>, Receiver<CollisionEvent>),
    event_handler: Box<rap2d::ChannelEventCollector>,
}

impl PhysicsWorld2D {
    pub fn new(gravity: Vector2<f32>) -> Self {
        let (sender, receiver) = unbounded();
        // The event collector bridges Rapier's hooks to Crossbeam channels
        let event_handler = Box::new(rap2d::ChannelEventCollector::new(sender.clone(), sender));

        Self {
            gravity,
            pipeline: rap2d::PhysicsPipeline::new(),
            integration_params: rap2d::IntegrationParameters::default(),
            islands: rap2d::IslandManager::new(),
            broad_phase: rap2d::BroadPhase::new(),
            narrow_phase: rap2d::NarrowPhase::new(),
            bodies: rap2d::RigidBodySet::new(),
            colliders: rap2d::ColliderSet::new(),
            impulse_joints: rap2d::ImpulseJointSet::new(),
            multibody_joints: rap2d::MultibodyJointSet::new(),
            ccd_solver: rap2d::CCDSolver::new(),
            query_pipeline: rap2d::QueryPipeline::new(),
            collision_events: (sender, receiver),
            event_handler,
        }
    }

    /// High-performance step function. 
    /// dt: delta time. 
    /// Returns a reference to collision events occurred this frame.
    pub fn step(&mut self, dt: f32) {
        self.integration_params.dt = dt;

        // Update the query pipeline *before* stepping if you need up-to-date queries
        // within the frame, or after if you need them for the next frame.
        // We do it here to ensure the query pipeline matches the new state.
        self.query_pipeline.update(&self.bodies, &self.colliders);

        self.pipeline.step(
            &self.gravity,
            &self.integration_params,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            Some(&mut self.query_pipeline),
            &*self.event_handler, // Pass the event hook
            &(), // No physics hooks user data
        );
    }

    /// Drains collision events from the channel. Non-blocking.
    pub fn drain_events(&self) -> Vec<CollisionEvent> {
        self.collision_events.1.try_iter().collect()
    }

    // -------------------------------------------------------------------------
    // Body Management
    // -------------------------------------------------------------------------

    /// Creates a rigid body with a box collider attached.
    /// Returns the RigidBodyHandle.
    pub fn spawn_box(
        &mut self,
        position: Vector2<f32>,
        rotation: f32,
        size: Vector2<f32>, // Half-extents
        body_type: BodyType,
        filter: Option<CollisionFilter>,
    ) -> rap2d::RigidBodyHandle {
        let builder = match body_type {
            BodyType::Dynamic => rap2d::RigidBodyBuilder::dynamic(),
            BodyType::Static => rap2d::RigidBodyBuilder::fixed(),
            BodyType::Kinematic => rap2d::RigidBodyBuilder::kinematic_velocity_based(),
        };

        let body = builder
            .translation(position)
            .rotation(rotation)
            .build();

        let mut collider = rap2d::ColliderBuilder::cuboid(size.x, size.y)
            .build();
            
        if let Some(f) = filter {
            collider.set_collision_groups(rap2d::InteractionGroups::new(f.membership, f.mask));
        }

        let handle = self.bodies.insert(body);
        self.colliders.insert_with_parent(collider, handle, &mut self.bodies);
        handle
    }

    /// Applies a force at center of mass.
    pub fn apply_force(&mut self, handle: rap2d::RigidBodyHandle, force: Vector2<f32>, wake_up: bool) {
        if let Some(body) = self.bodies.get_mut(handle) {
            body.apply_force(force, wake_up);
        }
    }

    /// Applies an impulse (instant velocity change).
    pub fn apply_impulse(&mut self, handle: rap2d::RigidBodyHandle, impulse: Vector2<f32>, wake_up: bool) {
        if let Some(body) = self.bodies.get_mut(handle) {
            body.apply_impulse(impulse, wake_up);
        }
    }

    /// Set linear velocity directly.
    pub fn set_velocity(&mut self, handle: rap2d::RigidBodyHandle, vel: Vector2<f32>) {
        if let Some(body) = self.bodies.get_mut(handle) {
            body.set_linvel(vel, true);
        }
    }

    /// Retrieves position and rotation.
    #[inline(always)]
    pub fn get_transform(&self, handle: rap2d::RigidBodyHandle) -> Option<(Vector2<f32>, f32)> {
        self.bodies.get(handle).map(|b| {
            let pos = b.position();
            (*pos.translation.vector, pos.rotation.angle())
        })
    }

    // -------------------------------------------------------------------------
    // Advanced Queries
    // -------------------------------------------------------------------------

    /// Raycast against the world.
    pub fn raycast(&self, origin: Point2<f32>, dir: Vector2<f32>, max_toi: f32, filter: Option<CollisionFilter>) -> Option<(rap2d::ColliderHandle, f32, Vector2<f32>)> {
        let dir = Unit::new_normalize(dir);
        let ray = rap2d::Ray::new(origin, *dir);
        
        let query_filter = filter.map(|f| {
            rap2d::QueryFilter::new().groups(rap2d::InteractionGroups::new(f.membership, f.mask))
        }).unwrap_or(rap2d::QueryFilter::default());

        if let Some((handle, toi)) = self.query_pipeline.cast_ray(
            &self.bodies, &self.colliders, &ray, max_toi, true, query_filter, None
        ) {
            let point = ray.point_at(toi);
            let normal = self.colliders.get(handle)?.shape().normal_at_point(&rap2d::Ray::new(point, *dir), point).unwrap_or(*dir);
            Some((handle, toi, normal))
        } else {
            None
        }
    }

    /// Shape casting (Swept test) - Essential for collision prediction.
    pub fn shapecast(
        &self, 
        shape: &rap2d::SharedShape, 
        start_pos: Isometry2<f32>, 
        dir: Vector2<f32>, 
        max_toi: f32
    ) -> Option<(rap2d::ColliderHandle, f32)> {
        let dir = Unit::new_normalize(dir);
        self.query_pipeline.cast_shape(
            &self.bodies, &self.colliders, &start_pos, &*dir, &*shape, max_toi, rap2d::QueryFilter::default(), None
        )
    }
    
    // -------------------------------------------------------------------------
    // Joints
    // -------------------------------------------------------------------------

    /// Creates a revolute joint (hinge) between two bodies.
    pub fn add_revolute_joint(
        &mut self,
        body1: rap2d::RigidBodyHandle,
        body2: rap2d::RigidBodyHandle,
        anchor1: Point2<f32>,
        anchor2: Point2<f32>,
    ) {
        let params = rap2d::RevoluteJointBuilder::new()
            .local_anchor1(anchor1)
            .local_anchor2(anchor2)
            .build();
        
        self.impulse_joints.insert(body1, body2, params, true);
    }
}

// ---------------------------------------------------------------------------
// 3D Physics Engine (Mirrors 2D functionality for high performance)
// ---------------------------------------------------------------------------

pub struct PhysicsWorld3D {
    gravity: Vector3<f32>,
    pipeline: rap3d::PhysicsPipeline,
    integration_params: rap3d::IntegrationParameters,
    islands: rap3d::IslandManager,
    broad_phase: rap3d::BroadPhase,
    narrow_phase: rap3d::NarrowPhase,
    bodies: rap3d::RigidBodySet,
    colliders: rap3d::ColliderSet,
    impulse_joints: rap3d::ImpulseJointSet,
    multibody_joints: rap3d::MultibodyJointSet,
    ccd_solver: rap3d::CCDSolver,
    query_pipeline: rap3d::QueryPipeline,
    collision_events: (Sender<CollisionEvent>, Receiver<CollisionEvent>),
    event_handler: Box<rap3d::ChannelEventCollector>,
}

impl PhysicsWorld3D {
    pub fn new(gravity: Vector3<f32>) -> Self {
        let (sender, receiver) = unbounded();
        let event_handler = Box::new(rap3d::ChannelEventCollector::new(sender.clone(), sender));
        
        Self {
            gravity,
            pipeline: rap3d::PhysicsPipeline::new(),
            integration_params: rap3d::IntegrationParameters::default(),
            islands: rap3d::IslandManager::new(),
            broad_phase: rap3d::BroadPhase::new(),
            narrow_phase: rap3d::NarrowPhase::new(),
            bodies: rap3d::RigidBodySet::new(),
            colliders: rap3d::ColliderSet::new(),
            impulse_joints: rap3d::ImpulseJointSet::new(),
            multibody_joints: rap3d::MultibodyJointSet::new(),
            ccd_solver: rap3d::CCDSolver::new(),
            query_pipeline: rap3d::QueryPipeline::new(),
            collision_events: (sender, receiver),
            event_handler,
        }
    }

    pub fn step(&mut self, dt: f32) {
        self.integration_params.dt = dt;
        self.query_pipeline.update(&self.bodies, &self.colliders);
        self.pipeline.step(
            &self.gravity,
            &self.integration_params,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            Some(&mut self.query_pipeline),
            &*self.event_handler,
            &(),
        );
    }

    pub fn drain_events(&self) -> Vec<CollisionEvent> {
        self.collision_events.1.try_iter().collect()
    }

    /// Spawn a generic box or sphere.
    pub fn spawn_body(
        &mut self,
        position: Vector3<f32>,
        rotation: Rotation3<f32>,
        shape: rap3d::SharedShape,
        body_type: BodyType,
        filter: Option<CollisionFilter>,
    ) -> rap3d::RigidBodyHandle {
        let builder = match body_type {
            BodyType::Dynamic => rap3d::RigidBodyBuilder::dynamic(),
            BodyType::Static => rap3d::RigidBodyBuilder::fixed(),
            BodyType::Kinematic => rap3d::RigidBodyBuilder::kinematic_velocity_based(),
        };

        let iso = Isometry3::from_parts(position.into(), rotation);
        let body = builder.position(iso).build();
        
        let mut collider = rap3d::ColliderBuilder::new(shape).build();
        if let Some(f) = filter {
            collider.set_collision_groups(rap3d::InteractionGroups::new(f.membership, f.mask));
        }

        let handle = self.bodies.insert(body);
        self.colliders.insert_with_parent(collider, handle, &mut self.bodies);
        handle
    }
    
    // Convenience wrapper for boxes
    pub fn spawn_box(&mut self, pos: Vector3<f32>, size: Vector3<f32>, bt: BodyType) -> rap3d::RigidBodyHandle {
        let shape = rap3d::SharedShape::cuboid(size.x, size.y, size.z);
        self.spawn_body(pos, Rotation3::identity(), shape, bt, None)
    }

    pub fn apply_impulse(&mut self, handle: rap3d::RigidBodyHandle, impulse: Vector3<f32>, wake_up: bool) {
        if let Some(body) = self.bodies.get_mut(handle) {
            body.apply_impulse(impulse, wake_up);
        }
    }

    pub fn get_transform(&self, handle: rap3d::RigidBodyHandle) -> Option<Isometry3<f32>> {
        self.bodies.get(handle).map(|b| *b.position())
    }
    
    /// Advanced: Shape casting for movement collision prediction.
    pub fn shapecast(
        &self, 
        shape: &rap3d::SharedShape, 
        start_pos: Isometry3<f32>, 
        dir: Vector3<f32>, 
        max_toi: f32
    ) -> Option<(rap3d::ColliderHandle, f32)> {
        let dir = Unit::new_normalize(dir);
        self.query_pipeline.cast_shape(
            &self.bodies, &self.colliders, &start_pos, &*dir, &*shape, max_toi, rap3d::QueryFilter::default(), None
        )
    }
    
    pub fn add_spherical_joint(
        &mut self,
        body1: rap3d::RigidBodyHandle,
        body2: rap3d::RigidBodyHandle,
        anchor1: Point3<f32>,
        anchor2: Point3<f32>,
    ) {
        let params = rap3d::SphericalJointBuilder::new()
            .local_anchor1(anchor1)
            .local_anchor2(anchor2)
            .build();
        self.impulse_joints.insert(body1, body2, params, true);
    }
}

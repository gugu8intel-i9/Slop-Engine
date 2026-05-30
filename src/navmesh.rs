// src/navmesh.rs
// Navigation mesh generation + pathfinding

use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;
use glam::{Vec3, Vec2};
use parking_lot::RwLock;

/// Navigation mesh polygon
#[derive(Clone, Debug)]
pub struct NavPolygon {
    pub id: u32,
    pub vertices: Vec<Vec3>,
    pub neighbors: Vec<Option<u32>>, // Neighbor polygon indices
    pub portal_edges: Vec<(usize, usize)>, // Edge indices to neighbors
    pub area_type: AreaType,
    pub center: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AreaType {
    Walkable,
    NotWalkable,
    Water,
    Road,
    Door,
    Custom(u8),
}

impl NavPolygon {
    pub fn new(id: u32, vertices: Vec<Vec3>) -> Self {
        let center = Self::compute_center(&vertices);
        Self {
            id,
            vertices,
            neighbors: Vec::new(),
            portal_edges: Vec::new(),
            area_type: AreaType::Walkable,
            center,
        }
    }

    fn compute_center(vertices: &[Vec3]) -> Vec3 {
        if vertices.is_empty() {
            return Vec3::ZERO;
        }
        let sum: Vec3 = vertices.iter().sum();
        sum / vertices.len() as f32
    }

    pub fn contains_point_xz(&self, point: Vec3) -> bool {
        // Point-in-polygon test on XZ plane
        let mut inside = false;
        let n = self.vertices.len();
        
        for i in 0..n {
            let j = (i + 1) % n;
            let vi = self.vertices[i];
            let vj = self.vertices[j];
            
            if ((vi.z > point.z) != (vj.z > point.z))
                && (point.x < (vj.x - vi.x) * (point.z - vi.z) / (vj.z - vi.z) + vi.x)
            {
                inside = !inside;
            }
        }
        
        inside
    }

    pub fn get_edge(&self, edge_index: usize) -> Option<(Vec3, Vec3)> {
        if edge_index < self.vertices.len() {
            let next = (edge_index + 1) % self.vertices.len();
            Some((self.vertices[edge_index], self.vertices[next]))
        } else {
            None
        }
    }
}

/// Navigation query result
#[derive(Clone, Debug)]
pub struct NavPath {
    pub polygons: Vec<u32>,
    pub waypoints: Vec<Vec3>,
    pub total_distance: f32,
}

/// A* search node
#[derive(Clone, Debug)]
struct PathNode {
    polygon_id: u32,
    g_cost: f32, // Cost from start
    h_cost: f32, // Heuristic cost to goal
    parent: Option<u32>,
    entry_point: Vec3,
}

impl PartialEq for PathNode {
    fn eq(&self, other: &Self) -> bool {
        self.polygon_id == other.polygon_id
    }
}

impl Eq for PathNode {}

impl PartialOrd for PathNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PathNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        let self_total = self.g_cost + self.h_cost;
        let other_total = other.g_cost + other.h_cost;
        other_total.partial_cmp(&self_total).unwrap_or(Ordering::Equal)
    }
}

/// Navigation mesh system
pub struct NavMesh {
    pub polygons: RwLock<HashMap<u32, NavPolygon>>,
    pub bounds: RwLock<(Vec3, Vec3)>, // min, max
    pub cell_size: f32,
    pub cell_height: f32,
}

impl NavMesh {
    pub fn new(cell_size: f32, cell_height: f32) -> Self {
        Self {
            polygons: RwLock::new(HashMap::new()),
            bounds: RwLock::new((Vec3::ZERO, Vec3::ZERO)),
            cell_size,
            cell_height,
        }
    }

    /// Add a polygon to the navmesh
    pub fn add_polygon(&self, mut polygon: NavPolygon) {
        let id = polygon.id;
        
        // Update bounds
        let mut bounds = self.bounds.write();
        for vertex in &polygon.vertices {
            bounds.0 = bounds.0.min(*vertex);
            bounds.1 = bounds.1.max(*vertex);
        }
        
        self.polygons.write().insert(id, polygon);
    }

    /// Build neighbor relationships between polygons
    pub fn build_connectivity(&self, threshold: f32) {
        let polygons = self.polygons.read();
        let polygon_ids: Vec<u32> = polygons.keys().cloned().collect();
        
        drop(polygons);
        
        for i in 0..polygon_ids.len() {
            let poly_a_id = polygon_ids[i];
            
            for j in (i + 1)..polygon_ids.len() {
                let poly_b_id = polygon_ids[j];
                
                if let Some(shared_edge) = self.find_shared_edge(poly_a_id, poly_b_id, threshold) {
                    let mut polygons = self.polygons.write();
                    
                    if let Some(poly_a) = polygons.get_mut(&poly_a_id) {
                        poly_a.neighbors.push(Some(poly_b_id));
                        poly_a.portal_edges.push(shared_edge.0);
                    }
                    
                    if let Some(poly_b) = polygons.get_mut(&poly_b_id) {
                        poly_b.neighbors.push(Some(poly_a_id));
                        poly_b.portal_edges.push(shared_edge.1);
                    }
                }
            }
        }
    }

    fn find_shared_edge(&self, poly_a_id: u32, poly_b_id: u32, threshold: f32) -> Option<((usize, usize), (usize, usize))> {
        let polygons = self.polygons.read();
        let poly_a = polygons.get(&poly_a_id)?;
        let poly_b = polygons.get(&poly_b_id)?;
        
        for i in 0..poly_a.vertices.len() {
            let edge_a = poly_a.get_edge(i)?;
            
            for j in 0..poly_b.vertices.len() {
                let edge_b = poly_b.get_edge(j)?;
                
                if self.edges_match(edge_a, edge_b, threshold) {
                    return Some(((i, (i + 1) % poly_a.vertices.len()), 
                                 (j, (j + 1) % poly_b.vertices.len())));
                }
            }
        }
        
        None
    }

    fn edges_match(&self, a: (Vec3, Vec3), b: (Vec3, Vec3), threshold: f32) -> bool {
        // Check if edges match (possibly reversed)
        let direct_match = (a.0.distance(b.0) < threshold && a.1.distance(b.1) < threshold);
        let reversed_match = (a.0.distance(b.1) < threshold && a.1.distance(b.0) < threshold);
        direct_match || reversed_match
    }

    /// Find the polygon containing a point
    pub fn find_polygon_at(&self, point: Vec3) -> Option<u32> {
        let polygons = self.polygons.read();
        
        for (id, poly) in polygons.iter() {
            if poly.contains_point_xz(point) {
                return Some(*id);
            }
        }
        
        None
    }

    /// Find nearest polygon to a point
    pub fn find_nearest_polygon(&self, point: Vec3) -> Option<u32> {
        let polygons = self.polygons.read();
        
        let mut nearest_id = None;
        let mut nearest_dist = f32::MAX;
        
        for (id, poly) in polygons.iter() {
            let dist = poly.center.distance(point);
            if dist < nearest_dist {
                nearest_dist = dist;
                nearest_id = Some(*id);
            }
        }
        
        nearest_id
    }

    /// Find path using A* algorithm
    pub fn find_path(&self, start: Vec3, end: Vec3) -> Option<NavPath> {
        let start_poly = self.find_polygon_at(start)?;
        let end_poly = self.find_polygon_at(end)?;
        
        if start_poly == end_poly {
            return Some(NavPath {
                polygons: vec![start_poly],
                waypoints: vec![start, end],
                total_distance: start.distance(end),
            });
        }
        
        let mut open_set = BinaryHeap::new();
        let mut came_from: HashMap<u32, (u32, Vec3)> = HashMap::new();
        let mut g_scores: HashMap<u32, f32> = HashMap::new();
        
        g_scores.insert(start_poly, 0.0);
        open_set.push(PathNode {
            polygon_id: start_poly,
            g_cost: 0.0,
            h_cost: self.heuristic(start_poly, end_poly),
            parent: None,
            entry_point: start,
        });
        
        while let Some(current) = open_set.pop() {
            if current.polygon_id == end_poly {
                return Some(self.reconstruct_path(&came_from, current.polygon_id, start, end));
            }
            
            let polygons = self.polygons.read();
            let current_poly = polygons.get(&current.polygon_id)?;
            
            for (neighbor_idx, neighbor_opt) in current_poly.neighbors.iter().enumerate() {
                if let Some(neighbor_id) = neighbor_opt {
                    let portal_edge_idx = current_poly.portal_edges[neighbor_idx];
                    let portal = current_poly.get_edge(portal_edge_idx)?;
                    let portal_center = (portal.0 + portal.1) * 0.5;
                    
                    let tentative_g = current.g_cost + current.entry_point.distance(portal_center);
                    
                    if tentative_g < *g_scores.get(neighbor_id).unwrap_or(&f32::MAX) {
                        came_from.insert(*neighbor_id, (current.polygon_id, portal_center));
                        g_scores.insert(*neighbor_id, tentative_g);
                        
                        open_set.push(PathNode {
                            polygon_id: *neighbor_id,
                            g_cost: tentative_g,
                            h_cost: self.heuristic(*neighbor_id, end_poly),
                            parent: Some(current.polygon_id),
                            entry_point: portal_center,
                        });
                    }
                }
            }
        }
        
        None
    }

    fn heuristic(&self, from: u32, to: u32) -> f32 {
        let polygons = self.polygons.read();
        let from_poly = polygons.get(&from);
        let to_poly = polygons.get(&to);
        
        if let (Some(a), Some(b)) = (from_poly, to_poly) {
            a.center.distance(b.center)
        } else {
            f32::MAX
        }
    }

    fn reconstruct_path(
        &self,
        came_from: &HashMap<u32, (u32, Vec3)>,
        mut current: u32,
        start: Vec3,
        end: Vec3,
    ) -> NavPath {
        let mut polygons = vec![current];
        let mut waypoints = vec![end];
        
        while let Some((parent, entry_point)) = came_from.get(&current) {
            polygons.push(*parent);
            waypoints.push(*entry_point);
            current = *parent;
        }
        
        polygons.push(start.into()); // Start polygon marker
        waypoints.push(start);
        
        polygons.reverse();
        waypoints.reverse();
        
        let total_distance = waypoints.windows(2)
            .map(|w| w[0].distance(w[1]))
            .sum();
        
        NavPath {
            polygons,
            waypoints,
            total_distance,
        }
    }

    /// Simple line-of-sight check between two points
    pub fn line_of_sight(&self, from: Vec3, to: Vec3) -> bool {
        let from_poly = match self.find_polygon_at(from) {
            Some(p) => p,
            None => return false,
        };
        
        let to_poly = match self.find_polygon_at(to) {
            Some(p) => p,
            None => return false,
        };
        
        // If same polygon, LOS is clear
        if from_poly == to_poly {
            return true;
        }
        
        // Check if path exists and is direct
        if let Some(path) = self.find_path(from, to) {
            // Simple check: if path has only 2 waypoints, it's direct
            return path.waypoints.len() <= 2;
        }
        
        false
    }

    /// Raycast on navmesh
    pub fn raycast(&self, origin: Vec3, direction: Vec3, max_distance: f32) -> Option<(Vec3, u32)> {
        let mut current_pos = origin;
        let mut remaining = max_distance;
        let mut current_poly = self.find_polygon_at(origin)?;
        
        while remaining > 0.0 {
            let polygons = self.polygons.read();
            let poly = polygons.get(&current_poly)?;
            
            // Find exit point from current polygon
            let ray_end = current_pos + direction * remaining;
            
            if poly.contains_point_xz(ray_end) {
                return Some((ray_end, current_poly));
            }
            
            // Find intersection with polygon edges
            for (neighbor_idx, neighbor_opt) in poly.neighbors.iter().enumerate() {
                if let Some(neighbor_id) = neighbor_opt {
                    let portal_edge_idx = poly.portal_edges[neighbor_idx];
                    if let Some((v1, v2)) = poly.get_edge(portal_edge_idx) {
                        if let Some(hit) = self.ray_segment_intersect(current_pos, direction, v1, v2) {
                            if hit.distance(current_pos) <= remaining {
                                current_pos = hit;
                                remaining -= current_pos.distance(hit);
                                current_poly = *neighbor_id;
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        Some((current_pos, current_poly))
    }

    fn ray_segment_intersect(&self, origin: Vec3, direction: Vec3, a: Vec3, b: Vec3) -> Option<Vec3> {
        // Simplified 2D ray-segment intersection on XZ plane
        let dir_xz = Vec2::new(direction.x, direction.z).normalize();
        let seg_a = Vec2::new(a.x, a.z);
        let seg_b = Vec2::new(b.x, b.z);
        let orig_xz = Vec2::new(origin.x, origin.z);
        
        let seg_dir = seg_b - seg_a;
        let denom = dir_xz.perp_dot(seg_dir);
        
        if denom.abs() < 0.0001 {
            return None;
        }
        
        let t = (orig_xz - seg_a).perp_dot(seg_dir) / denom;
        let u = (orig_xz - seg_a).perp_dot(dir_xz) / denom;
        
        if t >= 0.0 && u >= 0.0 && u <= 1.0 {
            let hit = orig_xz + dir_xz * t;
            Some(Vec3::new(hit.x, origin.y, hit.y))
        } else {
            None
        }
    }

    /// Get all polygons in an area
    pub fn get_polygons_in_radius(&self, center: Vec3, radius: f32) -> Vec<u32> {
        let polygons = self.polygons.read();
        let mut result = Vec::new();
        
        for (id, poly) in polygons.iter() {
            if poly.center.distance(center) <= radius {
                result.push(*id);
            }
        }
        
        result
    }

    /// Move a point along the surface
    pub fn move_on_surface(&self, position: Vec3, velocity: Vec3, dt: f32) -> Vec3 {
        if let Some(poly_id) = self.find_polygon_at(position) {
            let polygons = self.polygons.read();
            if let Some(poly) = polygons.get(&poly_id) {
                let target = position + velocity * dt;
                if poly.contains_point_xz(target) {
                    return target;
                }
            }
        }
        
        // Fall back to pathfinding
        let target = position + velocity * dt;
        if let Some(path) = self.find_path(position, target) {
            if path.waypoints.len() > 1 {
                return path.waypoints[1];
            }
        }
        
        position
    }
}

impl Default for NavMesh {
    fn default() -> Self {
        Self::new(0.3, 0.2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_navmesh_creation() {
        let navmesh = NavMesh::new(0.3, 0.2);
        
        let poly = NavPolygon::new(0, vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
        ]);
        
        navmesh.add_polygon(poly);
        
        assert!(navmesh.find_polygon_at(Vec3::new(0.5, 0.0, 0.5)).is_some());
    }

    #[test]
    fn test_point_in_polygon() {
        let poly = NavPolygon::new(0, vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, 2.0),
        ]);
        
        assert!(poly.contains_point_xz(Vec3::new(1.0, 0.0, 1.0)));
        assert!(!poly.contains_point_xz(Vec3::new(3.0, 0.0, 3.0)));
    }
}

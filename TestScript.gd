@tool
class_name objectScatter extends Marker3D

@export_flags_3d_physics var COLLISION_LAYER_HIT : int = 65536
@export_flags_3d_physics var COLLISION_LAYER_EXCLUDE : int = 131072

@export var compute_multimesh_target : GPUComputeMultimesh3D

@export var object_count : int = 100
@export var max_distance : float = 10.0
@export var coverage_area : Vector2 = Vector2.ONE

@export var scatterSeed : int = 0

@export var align_to_surface_normal : bool = false

@export var base_size : Vector3 = Vector3.ONE
@export var size_variation : float = 0.0

@export var staticBody : StaticBody3D
@export var collider : Shape3D

@export_tool_button("Run") var runButton = _init
@export_tool_button("Clear") var clearCompute = _clear_objects

@export_range(0.0, 360.0) var y_variation : float = 0.0

var hitResults : Array[Array]
var transforms : Array[Transform3D]
var RNG : RandomNumberGenerator
var completed : bool = false


func _ready() -> void:
	compute_multimesh_target.clear_compute()
	await get_tree().create_timer(0.1).timeout
	_init()

func _clear_objects() -> void:
	compute_multimesh_target.clear_compute()

func _init():
	RNG = RandomNumberGenerator.new()
	RNG.set_seed(scatterSeed)
	
	completed = false
	hitResults.clear()
	transforms.clear()
	
	if !compute_multimesh_target:
		#printerr("No compute multimesh assigned")
		return
	
	for i in range(object_count):
		hitResults.append(_random_raycast())
	
	if staticBody:
		for child in staticBody.get_children():
			child.queue_free()
	
	for result : Array in hitResults: #[0] = 
		if result[0] == true:
			var newBasis : Basis = Basis.IDENTITY
			
			var newTransform : Transform3D = Transform3D(newBasis, result[1])
			
			if align_to_surface_normal:
				newTransform = newTransform.looking_at(result[1] + Vector3(result[2]).cross(Vector3.RIGHT))
			
			newTransform = newTransform.scaled_local(base_size + base_size * RNG.randf_range(-size_variation/2.0, size_variation/2.0))
			
			newTransform = newTransform.rotated_local(Vector3.UP, RNG.randf_range(-y_variation/2.0, y_variation/2.0) * PI/180.0)
			
			#newTransform = newTransform.orthonormalized()
			
			#print(newTransform)
			
			if newTransform.origin == Vector3.ZERO:
				printerr("Position at world origin")
			
			transforms.append(newTransform)
			
			if staticBody:
				var newCollisionShape : CollisionShape3D = CollisionShape3D.new()
				staticBody.add_child(newCollisionShape)
				newCollisionShape.set_owner(owner)
				newCollisionShape.shape = collider
				newCollisionShape.transform = newTransform
				
	
	if transforms.size() > 0:
		compute_multimesh_target.ingest_transforms(transforms)
		compute_multimesh_target.initialize_compute()
		compute_multimesh_target.update_visible()
		compute_multimesh_target.iterate_frustum_culling = true
	else:
		print(self.name, " not enough targets hit...")

#func _unhandled_input(_event: InputEvent) -> void:
	#if Input.is_action_just_pressed("DEBUG_FOLIAGE_REBUILD"):
		#_init()

func _random_raycast() -> Array:
	
	var randomPosition : Vector3 = global_position
	
	
	
	randomPosition.x += RNG.randf_range(-coverage_area.x/2.0, coverage_area.x/2.0)
	randomPosition.z += RNG.randf_range(-coverage_area.y/2.0, coverage_area.y/2.0)
	
	var hitRaycast : RayCast3D
	var excludeRaycast : RayCast3D
	
	excludeRaycast = RayCast3D.new()
	add_child(excludeRaycast)
	excludeRaycast.global_position = randomPosition
	excludeRaycast.target_position = -Vector3.UP * max_distance
	excludeRaycast.collision_mask = COLLISION_LAYER_EXCLUDE
	excludeRaycast.force_raycast_update()
	
	if excludeRaycast.is_colliding():
		excludeRaycast.queue_free()
		return [false, Vector3.ZERO, Vector3.UP]
	
	excludeRaycast.queue_free()
	
	hitRaycast = RayCast3D.new()
	add_child(hitRaycast)
	hitRaycast.global_position = randomPosition
	hitRaycast.target_position = -Vector3.UP * max_distance
	hitRaycast.collision_mask = COLLISION_LAYER_HIT
	hitRaycast.force_raycast_update()
	
	var NORMAL : Vector3 = Vector3.UP
	
	if hitRaycast.is_colliding():
		NORMAL = hitRaycast.get_collision_normal()
	
	if hitRaycast.is_colliding() and NORMAL != Vector3.ZERO:
		hitRaycast.queue_free()
		return [true, hitRaycast.get_collision_point(), NORMAL]
	else:
		hitRaycast.queue_free()
		return [false, Vector3.ZERO, Vector3.UP]

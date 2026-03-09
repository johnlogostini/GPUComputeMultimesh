@tool
extends Node3D
class_name GPUComputeMultimesh3D

static var foliage_frustum_invocation_count: int = 32

@export_tool_button("Initialize") var initialize = initialize_compute
@export_tool_button("IngestMultimesh") var ingest = ingest_multimesh

@export var ingested_multimesh: MultiMeshInstance3D

@export var iterate_frustum_culling: bool = false

@export var instance_fade_speed: float = 1.0
@export var camera_fade_multiplier: float = 10.0
@export var instance_count: int = 0
@export_range(1, 64, 1) var foliage_instances_per_frustum_invocation: int = 4
@export var instance_frustum_bias: float = 2.0:
	get:
		return instance_frustum_bias
	set(value):
		instance_frustum_bias = value
		update_foliage_variables()

@export var sun_instance_frustum_bias: float = 2.0:
	get:
		return sun_instance_frustum_bias
	set(value):
		sun_instance_frustum_bias = value
		update_foliage_variables()

@export_range(0.1, 3.0) var lod_multiplier: float = 1.0:
	get:
		return lod_multiplier
	set(value):
		lod_multiplier = value
		update_foliage_variables()

@export var enable_dynamic_gi: bool = false:
	get:
		return enable_dynamic_gi
	set(value):
		enable_dynamic_gi = value
		update_foliage_variables()

@export var instance_data: PackedFloat32Array
@export var mesh_lods: Dictionary[Mesh, Vector2] = {}:
	get:
		return mesh_lods
	set(value):
		mesh_lods = value
		update_foliage_variables()

@export var mesh_shadows: Array[bool] = []:
	get:
		return mesh_shadows
	set(value):
		mesh_shadows = value
		update_foliage_variables()
@export_flags_3d_render var rendered_layers: int = 0

@export var target_camera: Camera3D
@export var sun: DirectionalLight3D

@export var foliage_frustum_culling_shader: RDShaderFile = preload("res://addons/gpucomputemultimesh3d/FoliageFrustumCulling.glsl")

#region Compute Core
class WanderTerrainComputeBlock:
	var pipeline: RID = RID()
	var cur_shader: RID = RID()
	var uniform_set: RID = RID()
	var initialized: bool = false
	var pre_built: bool = false
	
	var prepass_uniforms_array: Array[RDUniform] = []
	
	var internal_invocations: Vector3i
	
	#region uniforms
	func begin_uniforms(rd: RenderingDevice):
		prepass_uniforms_array = []
		if uniform_set.is_valid():
			rd.free_rid(uniform_set)
		uniform_set = RID()
	
	func push_uniform_array(uniform_type: RenderingDevice.UniformType, base_rid: Array[RID]):
		var new_uniform = RDUniform.new()
		new_uniform.uniform_type = uniform_type
		new_uniform.binding = prepass_uniforms_array.size()
		
		for rid in base_rid:
			new_uniform.add_id(rid)
		
		
		prepass_uniforms_array.append(new_uniform)
	
	# if a rid is managed by the block, this means it will be cleaned up when the block is cleaned.
	# certain textures are used by multiple blocks, like the atlas, as such it should only be managed by one block.
	func push_uniform(uniform_type: RenderingDevice.UniformType, base_rid: RID, sampler_rid: RID = RID()):
		var new_uniform = RDUniform.new()
		new_uniform.uniform_type = uniform_type
		new_uniform.binding = prepass_uniforms_array.size()
		if sampler_rid.is_valid():
			new_uniform.add_id(sampler_rid)
		new_uniform.add_id(base_rid)
		
		prepass_uniforms_array.append(new_uniform)
	
	func finish_uniforms(rd: RenderingDevice):
		uniform_set = rd.uniform_set_create(prepass_uniforms_array, cur_shader, 0)
	#endregion
	
	#region basic
	func setup_compute(rd: RenderingDevice, shader: RID, new_internal_invocations: Vector3i):
		internal_invocations = new_internal_invocations
		#if cur_shader.is_valid():
			#rd.free_rid(cur_shader)
		#
		cur_shader = shader
		if cur_shader.is_valid():
			pipeline = rd.compute_pipeline_create(shader)
			initialized = true
		else:
			printerr("Post pass Shader failed to compile.")
	
	func clear_compute(rd: RenderingDevice):
		#if cur_shader.is_valid():
			#rd.free_rid(cur_shader)
		cur_shader = RID()
		uniform_set = RID()
		initialized = false
	
	func execute_compute(rd: RenderingDevice, invocationgroups: Vector3i):
		var compute_list = rd.compute_list_begin()
		rd.compute_list_bind_compute_pipeline(compute_list, pipeline)
		rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
		rd.compute_list_dispatch(compute_list, invocationgroups.x, invocationgroups.y, invocationgroups.z)
		rd.compute_list_end()
		
		#WanderTerrain.invocations_per_frame_count += ((invocationgroups.x * internal_invocations.x) * (invocationgroups.y * internal_invocations.y)) * (invocationgroups.z * internal_invocations.z)
	#endregion

class FoliageComputeStruct:
	var foliage_iteration_block: WanderTerrainComputeBlock = WanderTerrainComputeBlock.new()
	var foliage_mesh: Mesh
	
	var cur_invocation_count: int = 1
	var cur_instance_count: int = 1
	var instance_storage_buffer: RID = RID()
	var instance_chunk_data_buffer: RID = RID() #Contains the max count, as well as the position and scale of the chunk, for frustum culling
	var instance_chunk_data: PackedByteArray = PackedByteArray()
	
	var multimesh_rs_rid: RID = RID() #only render server, remove using renderserver
	var multimesh_rs_instance_rid: RID = RID() #only render server, remove using renderserver
	var command_buffer: RID = RID()
	var multimesh_data_buffer: RID = RID()
	
	func clear_compute(rd: RenderingDevice):
		if foliage_iteration_block && foliage_iteration_block.initialized:
			foliage_iteration_block.clear_compute(rd)
		
		if instance_storage_buffer.is_valid():
			rd.free_rid(instance_storage_buffer)
		instance_storage_buffer = RID()
		
		if instance_chunk_data_buffer.is_valid():
			rd.free_rid(instance_chunk_data_buffer)
		instance_chunk_data_buffer = RID()
		
		if multimesh_rs_rid.is_valid():
			RenderingServer.free_rid(multimesh_rs_rid)
		multimesh_rs_rid = RID()
		
		if multimesh_rs_instance_rid.is_valid():
			RenderingServer.free_rid(multimesh_rs_instance_rid)
		multimesh_rs_instance_rid = RID()
		
		if multimesh_data_buffer.is_valid():
			RenderingServer.free_rid(multimesh_data_buffer)
		multimesh_data_buffer = RID()
		
		if command_buffer.is_valid():
			RenderingServer.free_rid(command_buffer)
		command_buffer = RID()
	
	func iterate_compute(rd: RenderingDevice, camera_data: RID):
		if cur_instance_count > 0:
			if !foliage_iteration_block.pre_built:
				foliage_iteration_block.pre_built = true
				foliage_iteration_block.begin_uniforms(rd)
				foliage_iteration_block.push_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, camera_data)
				
				foliage_iteration_block.push_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, instance_chunk_data_buffer)
				foliage_iteration_block.push_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, instance_storage_buffer)
				
				foliage_iteration_block.push_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, multimesh_data_buffer)
				foliage_iteration_block.push_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, command_buffer)
				
				foliage_iteration_block.finish_uniforms(rd)
			
			rd.draw_command_begin_label("Foliage Frustum Culling", Color.GREEN)
			foliage_iteration_block.execute_compute(rd, Vector3i(cur_invocation_count, 1, 1))
			rd.draw_command_end_label()
	
	func UpdateDelta(rd: RenderingDevice, delta: float):
		instance_chunk_data.encode_float(28, delta)
		rd.buffer_update(instance_chunk_data_buffer, 0,instance_chunk_data.size(), instance_chunk_data)
	
	func UpdateLODMultiplier(rd: RenderingDevice, instance_frustum_bias: float, sun_instance_frustum_bias: float,  lod_multiplier: float, use_shadows: bool, use_dynamic_gi: bool, view_range: Vector2):
		instance_chunk_data.encode_float(20, sun_instance_frustum_bias)
		instance_chunk_data.encode_float(24, instance_frustum_bias)
		instance_chunk_data.encode_float(32, lod_multiplier)
		instance_chunk_data.encode_float(40, view_range.x)
		instance_chunk_data.encode_float(44, view_range.y)
		
		rd.buffer_update(instance_chunk_data_buffer, 0,instance_chunk_data.size(), instance_chunk_data)
		
		RenderingServer.instance_geometry_set_flag(multimesh_rs_instance_rid, RenderingServer.INSTANCE_FLAG_USE_DYNAMIC_GI, use_dynamic_gi)
		RenderingServer.instance_geometry_set_cast_shadows_setting(multimesh_rs_instance_rid, RenderingServer.SHADOW_CASTING_SETTING_ON if use_shadows else RenderingServer.SHADOW_CASTING_SETTING_OFF)
	
	static func Create(rd: RenderingDevice, world_scenario: RID, render_layers: int, foliage_per_workgroup: int, foliage_count: int, invocation_count: int, view_range: Vector2, instance_frustum_bias: float, sun_instance_frustum_bias: float, lod_multiplier: float, is_lowest_lod: bool, mesh: Mesh, use_shadows: bool, use_dynamic_gi: bool, foliage_compute: RDShaderSPIRV) -> FoliageComputeStruct:
		
		var new_foliage_compute: FoliageComputeStruct = FoliageComputeStruct.new()
		new_foliage_compute.foliage_mesh = mesh
		new_foliage_compute.cur_invocation_count = invocation_count
		new_foliage_compute.cur_instance_count = foliage_count
		new_foliage_compute.foliage_iteration_block.setup_compute(rd, rd.shader_create_from_spirv(foliage_compute), Vector3i(32,1,1))
		new_foliage_compute.instance_storage_buffer = rd.storage_buffer_create(foliage_count * 20 * 4)
		
		new_foliage_compute.instance_chunk_data.resize(48)
		
		#int total_foliage_instance_count;
		new_foliage_compute.instance_chunk_data.encode_s32(0, foliage_count)
		#int per_workgroup_instance_count;
		new_foliage_compute.instance_chunk_data.encode_s32(4, invocation_count)
		#int workgroup_count;
		new_foliage_compute.instance_chunk_data.encode_s32(8, mesh.get_surface_count())
		#int mesh_surface_count;
		new_foliage_compute.instance_chunk_data.encode_s32(12, 0)
		
		#float instance_frustum_bias;
		new_foliage_compute.instance_chunk_data.encode_s32(16, 0)
		#float reserved;
		new_foliage_compute.instance_chunk_data.encode_float(20, sun_instance_frustum_bias)
		#uint current_counter;
		new_foliage_compute.instance_chunk_data.encode_float(24, instance_frustum_bias)
		#uint done_counter;
		new_foliage_compute.instance_chunk_data.encode_float(28, 0.05)
#
		#vec2 chunk_position_world;
		new_foliage_compute.instance_chunk_data.encode_float(32, lod_multiplier)
		new_foliage_compute.instance_chunk_data.encode_s32(36, 1 if is_lowest_lod else 0)
		#vec2 view_range;
		new_foliage_compute.instance_chunk_data.encode_float(40, view_range.x)
		new_foliage_compute.instance_chunk_data.encode_float(44, view_range.y)
		
		
		new_foliage_compute.instance_chunk_data_buffer = rd.storage_buffer_create(48, new_foliage_compute.instance_chunk_data)
		#print(rd.buffer_get_data(new_foliage_compute.instance_chunk_data_buffer, 0, new_foliage_compute.instance_chunk_data.size()).to_int32_array())
		#build the multimesh
		new_foliage_compute.multimesh_rs_rid = RenderingServer.multimesh_create()
		RenderingServer.multimesh_allocate_data(new_foliage_compute.multimesh_rs_rid, foliage_count, RenderingServer.MULTIMESH_TRANSFORM_3D, true, true, true)
		RenderingServer.multimesh_set_mesh(new_foliage_compute.multimesh_rs_rid, mesh)
		
		
		new_foliage_compute.multimesh_rs_instance_rid = RenderingServer.instance_create2(new_foliage_compute.multimesh_rs_rid, world_scenario)
		
		RenderingServer.instance_set_ignore_culling(new_foliage_compute.multimesh_rs_instance_rid, true)
		RenderingServer.instance_set_custom_aabb(new_foliage_compute.multimesh_rs_instance_rid, AABB( Vector3.ONE * -25000.0, Vector3.ONE * 25000.0))
		
		#instance data
		RenderingServer.instance_set_layer_mask(new_foliage_compute.multimesh_rs_instance_rid, render_layers)
		RenderingServer.instance_geometry_set_flag(new_foliage_compute.multimesh_rs_instance_rid, RenderingServer.INSTANCE_FLAG_USE_DYNAMIC_GI, use_dynamic_gi)
		RenderingServer.instance_geometry_set_flag(new_foliage_compute.multimesh_rs_instance_rid, RenderingServer.INSTANCE_FLAG_USE_BAKED_LIGHT, use_dynamic_gi)
		RenderingServer.instance_geometry_set_cast_shadows_setting(new_foliage_compute.multimesh_rs_instance_rid, RenderingServer.SHADOW_CASTING_SETTING_ON if use_shadows else RenderingServer.SHADOW_CASTING_SETTING_OFF)
		
		RenderingServer.instance_set_extra_visibility_margin(new_foliage_compute.multimesh_rs_instance_rid, 100000000.0)
		
		new_foliage_compute.command_buffer = RenderingServer.multimesh_get_command_buffer_rd_rid(new_foliage_compute.multimesh_rs_rid)
		new_foliage_compute.multimesh_data_buffer = RenderingServer.multimesh_get_buffer_rd_rid(new_foliage_compute.multimesh_rs_rid)
		
		return new_foliage_compute
#endregion

var _is_visible: bool = false
var _foliage_frustum_update_timer: float = 0.0

var _used_camera: Camera3D
var _last_camera_transform: Transform3D
var _rd: RenderingDevice
var _compute_initialized: bool = false
var _active_foliage: Array[FoliageComputeStruct] = []

var _current_camera_data_buffer: RID = RID()
var _current_camera_data_bytes: PackedByteArray = []

var _camera_last_position: Vector3
var _camera_velocity: float = 0.0

func _ready() -> void:
	visibility_changed.connect(update_visible)

func _enter_tree() -> void:
	if Engine.is_editor_hint():
		initialize_compute()
		update_visible()
		_foliage_frustum_update_timer = 1.0

func _exit_tree() -> void:
	clear_compute()

func ingest_multimesh() -> void:
	if !_rd:
		_rd = RenderingServer.get_rendering_device()
	
	instance_data.clear()
	instance_count = 0
	
	if !ingested_multimesh || !ingested_multimesh.multimesh:
		printerr("Multimesh must be provided to import data.")
		return
	
	#if !ingested_multimesh.multimesh.use_colors || !ingested_multimesh.multimesh.use_custom_data:
		#printerr("GPU compute multimeshes are designed to use the custom data and color attributes within multimeshes, please set this to true on the multimesh selected to ingest.")
		#return
	
	#if Engine.is_editor_hint():
		#print(EditorInterface.get_editor_viewport_3d().get_camera_3d().global_position)
	
	instance_count = ingested_multimesh.multimesh.instance_count
	var multimesh_rd = ingested_multimesh.multimesh.get_rid()
	var new_instance_data = RenderingServer.multimesh_get_buffer(multimesh_rd)
	var stride_count: int = 20
	if !ingested_multimesh.multimesh.use_colors:
		stride_count -= 4
	if !ingested_multimesh.multimesh.use_custom_data:
		stride_count -= 4
	
	for i in range(instance_count):
		var idx = i * stride_count
		instance_data.append(new_instance_data[idx]); idx += 1
		instance_data.append(new_instance_data[idx]); idx += 1
		instance_data.append(new_instance_data[idx]); idx += 1
		instance_data.append(new_instance_data[idx]); idx += 1
		
		instance_data.append(new_instance_data[idx]); idx += 1
		instance_data.append(new_instance_data[idx]); idx += 1
		instance_data.append(new_instance_data[idx]); idx += 1
		instance_data.append(new_instance_data[idx]); idx += 1
		
		instance_data.append(new_instance_data[idx]); idx += 1
		instance_data.append(new_instance_data[idx]); idx += 1
		instance_data.append(new_instance_data[idx]); idx += 1
		instance_data.append(new_instance_data[idx]); idx += 1
		
		instance_data.append(1.0)
		instance_data.append(1.0)
		instance_data.append(1.0)
		instance_data.append(0.0)
		
		instance_data.append(0.0)
		instance_data.append(0.0)
		instance_data.append(0.0)
		instance_data.append(0.0)

func ingest_transforms(transforms: Array[Transform3D]) -> void:
	instance_count = transforms.size()
	instance_data.clear()
	for trans in transforms:
		instance_data.append(trans.basis[0][0])
		instance_data.append(trans.basis[1][0])
		instance_data.append(trans.basis[2][0])
		instance_data.append(trans.origin[0])
		
		instance_data.append(trans.basis[0][1])
		instance_data.append(trans.basis[1][1])
		instance_data.append(trans.basis[2][1])
		instance_data.append(trans.origin[1])
		
		instance_data.append(trans.basis[0][2])
		instance_data.append(trans.basis[1][2])
		instance_data.append(trans.basis[2][2])
		instance_data.append(trans.origin[2])
		
		instance_data.append(1.0)
		instance_data.append(1.0)
		instance_data.append(1.0)
		instance_data.append(0.0)
		
		instance_data.append(0.0)
		instance_data.append(0.0)
		instance_data.append(0.0)
		instance_data.append(0.0)

func clear_compute() -> void:
	_rd = RenderingServer.get_rendering_device()
	if !_rd:
		return
	
	for layer in _active_foliage:
		layer.clear_compute(_rd)
	_active_foliage.clear()
	
	if _current_camera_data_buffer.is_valid():
		_rd.free_rid(_current_camera_data_buffer)
	_current_camera_data_buffer = RID()
	_compute_initialized = false

func initialize_compute() -> void:
	clear_compute()
	if !_rd:
		printerr("RD not found.")
		return
	
	if instance_count <= 0 || instance_data.size() == 0 || mesh_lods.size() == 0:
		printerr("No instances or meshes.")
		return
	
	_current_camera_data_bytes.clear()
	_current_camera_data_bytes.resize(((6 * 4) + 8) * 4)
	
	_current_camera_data_buffer = _rd.storage_buffer_create(_current_camera_data_bytes.size(), _current_camera_data_bytes)
	
	var invocation_count: int = maxi(ceilf(float(instance_count) / float(foliage_instances_per_frustum_invocation) / float(GPUComputeMultimesh3D.foliage_frustum_invocation_count)), 1)
	var foliage_per_workgroup: int = ceilf(float(instance_count) * float(GPUComputeMultimesh3D.foliage_frustum_invocation_count))
	
	var foliage_spirv = foliage_frustum_culling_shader.get_spirv()
	if mesh_shadows.size() != mesh_lods.size():
		mesh_shadows.resize(mesh_lods.size())
	
	print(invocation_count)
	var index: int = 0
	for mesh in mesh_lods:
		var new_compute = FoliageComputeStruct.Create(_rd, get_world_3d().scenario, rendered_layers, foliage_per_workgroup, instance_count, invocation_count, mesh_lods[mesh], instance_frustum_bias, sun_instance_frustum_bias, lod_multiplier, index == mesh_lods.size() - 1, mesh, mesh_shadows[index], enable_dynamic_gi, foliage_spirv)
		_rd.buffer_update(new_compute.instance_storage_buffer, 0, instance_data.size() * 4, instance_data.to_byte_array())
		_active_foliage.append(new_compute)
		index += 1
	
	_compute_initialized = true

func _process(delta: float) -> void:
	if iterate_frustum_culling && visible && _compute_initialized:
		iterate_camera()
		
		if _foliage_frustum_update_timer > 0.0:
			#print(clearing_foliage_chunks.size(), " - ", current_foliage_chunks.size())
			
			_foliage_frustum_update_timer -= minf(delta * (instance_fade_speed * _camera_velocity), 1.0)
			RenderingServer.call_on_render_thread(foliage_frustum_iteration.bind(minf(delta / (instance_fade_speed / _camera_velocity), 1.0)))

func update_foliage_variables():
	if _compute_initialized:
		var index: int = 0
		for foliage in _active_foliage:
			foliage.UpdateLODMultiplier(_rd, instance_frustum_bias, sun_instance_frustum_bias, lod_multiplier, mesh_shadows[index], enable_dynamic_gi, mesh_lods[foliage.foliage_mesh])
			index += 1
		
		_foliage_frustum_update_timer = 1.0

func foliage_frustum_iteration(delta: float) -> void:
	for foliage in _active_foliage:
		foliage.UpdateDelta(_rd, delta)
		foliage.iterate_compute(_rd, _current_camera_data_buffer)

func update_visible() -> void:
	for foliage in _active_foliage:
		RenderingServer.instance_set_visible(foliage.multimesh_rs_instance_rid, visible)

func iterate_camera() -> void:
	if Engine.is_editor_hint():
		_used_camera = EditorInterface.get_editor_viewport_3d().get_camera_3d()
	else:
		if is_instance_valid(target_camera):
			_used_camera = target_camera
		else:
			if !is_instance_valid(_used_camera):
				_used_camera = get_viewport().get_camera_3d()
	
	if !is_instance_valid(_used_camera):
		printerr("No Camera found.")
		iterate_frustum_culling = false
		return
	
	if instance_count <= 0:
		return
	
	if _used_camera.global_transform != _last_camera_transform:
		_last_camera_transform = _used_camera.global_transform
		update_camera_buffer(_used_camera)

func update_camera_buffer(camera: Camera3D) -> void:
	_camera_velocity = max(camera.global_position.distance_to(_camera_last_position) * camera_fade_multiplier, 1.0)
	_camera_last_position = camera.global_position
	
	var camera_frustum = camera.get_frustum()
	var _index: int = 0;
	
	for plane in camera_frustum:
		_current_camera_data_bytes.encode_float(_index, plane.x); _index += 4
		_current_camera_data_bytes.encode_float(_index, plane.y); _index += 4
		_current_camera_data_bytes.encode_float(_index, plane.z); _index += 4
		_current_camera_data_bytes.encode_float(_index, plane.d); _index += 4
	
	_current_camera_data_bytes.encode_float(_index, camera.global_position.x); _index += 4
	_current_camera_data_bytes.encode_float(_index, camera.global_position.y); _index += 4
	_current_camera_data_bytes.encode_float(_index, camera.global_position.z); _index += 4
	_current_camera_data_bytes.encode_float(_index, 0.0); _index += 4
	
	_current_camera_data_bytes.encode_float(_index, sun.global_transform.basis.z.x); _index += 4
	_current_camera_data_bytes.encode_float(_index, sun.global_transform.basis.z.y); _index += 4
	_current_camera_data_bytes.encode_float(_index, sun.global_transform.basis.z.z); _index += 4
	_current_camera_data_bytes.encode_float(_index, 0.0); _index += 4
	
	#if _current_camera_data_buffer.is_valid():
		#_rd.free_rid(_current_camera_data_buffer)
	#_current_camera_data_buffer = _rd.storage_buffer_create(_current_camera_data_bytes.size(), _current_camera_data_bytes)
	_rd.buffer_update(_current_camera_data_buffer, 0, _current_camera_data_bytes.size(), _current_camera_data_bytes)
	#RenderingServer.call_on_render_thread(foliage_frustum_iteration)
	#iterate_frustum_culling = false
	_foliage_frustum_update_timer = 1.0

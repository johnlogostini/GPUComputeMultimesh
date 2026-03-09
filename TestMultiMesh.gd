@tool
extends MultiMeshInstance3D

@export_tool_button("TestButton") var test = test_build


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass


func test_build() -> void:
	multimesh.buffer.clear()
	multimesh.instance_count = 100
	var stride_count: int = 20
	if !multimesh.use_colors:
		stride_count -= 4
	if !multimesh.use_custom_data:
		stride_count -= 4
	
	
	multimesh.buffer.resize(100 * stride_count)
	for i in range(100):
		var new_transform = Transform3D.IDENTITY
		new_transform.origin.x = i * 2.0
		var idx: int = i * stride_count
		multimesh.buffer[idx + 0] = new_transform.basis[0][0]
		multimesh.buffer[idx + 1] = new_transform.basis[1][0]
		multimesh.buffer[idx + 2] = new_transform.basis[2][0]
		multimesh.buffer[idx + 3] = new_transform.origin[0]
		
		multimesh.buffer[idx + 4] = new_transform.basis[0][1]
		multimesh.buffer[idx + 5] = new_transform.basis[1][1]
		multimesh.buffer[idx + 6] = new_transform.basis[2][1]
		multimesh.buffer[idx + 7] = new_transform.origin[1]
		
		multimesh.buffer[idx + 8] = new_transform.basis[0][2]
		multimesh.buffer[idx + 9] = new_transform.basis[1][2]
		multimesh.buffer[idx + 10] = new_transform.basis[2][2]
		multimesh.buffer[idx + 11] = new_transform.origin[2]
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		

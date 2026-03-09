#[compute]
#version 450

#define LOCALSIZEX 32

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) restrict buffer CameraData { 
    vec4 FrustumPlanes[6]; 
    vec4 PlayerPosition;
    vec4 SunDirection;
    } camera_data;

layout(std430, binding = 1) restrict buffer InstanceChunkData { 
    int total_foliage_instance_count;
    int workgroup_count;
    int mesh_surface_count;
    uint current_counter;

    uint done_counter;
    float sun_instance_frustum_bias;
    float instance_frustum_bias;
    float delta_time;

    float lod_bias;
    int lowest_lod;
    vec2 view_range;
} chunk_data;

layout(std430, binding = 2) restrict buffer FoliageReferenceData { float data[]; } reference_instance;

layout(std430, binding = 3) restrict buffer FoliageOutputData { float data[]; } output_instance;
layout(std430, binding = 4) restrict buffer FoliageCommandBuffer { int data[]; } command_buffer;


ivec3 getInvocationRange(int invocation_index, int total_count, int total_invocation_count){
    int instancesperthread = int(ceil(float(total_count) / float(total_invocation_count * LOCALSIZEX)));

    int starting_instance = invocation_index * instancesperthread;

    return ivec3(starting_instance, starting_instance + instancesperthread, instancesperthread);
}

int isInsideFrustum(vec3 position, float instanceSize) {
	for (int i = 0; i < 6; i++) {
		vec4 plane = camera_data.FrustumPlanes[i];
		if (dot(plane.xyz, position) - instanceSize > plane.w) {
			return 1;
		}
	}
	return 0;
}

int isInSunLine(vec3 instance_position, vec3 player_position, vec3 sun_direction, float bias){
    vec3 a = player_position;
    vec3 b = player_position + sun_direction * 10000.0;
    vec3 p = instance_position;

    vec3 ab = b - a;
    float t = dot(p - a, ab) / dot(ab, ab);
    t = clamp(t, 0.0, 1.0);
    return int(length(instance_position - (a + t * ab)) > bias);
}

shared float sKeep[1024];
shared uint sOrigins[1024];
shared uint sDestinations[1024];
shared uint sBase;   // base output offset for this batch
shared uint sTotalKept;

void main() {
    uint lane = gl_LocalInvocationID.x;
    uint lane_count = gl_WorkGroupSize.x;
    uint global_invocation_thread = gl_GlobalInvocationID.x;
    // uint group = gl_WorkGroupID.x;
    

    ivec3 range = getInvocationRange(int(global_invocation_thread), chunk_data.total_foliage_instance_count, chunk_data.workgroup_count); //x = starting index, y = ending index, z = instances count per thread
    // if (range.x < 0) return;

    vec2 ViewDistance = chunk_data.view_range * chunk_data.lod_bias;
    vec3 playerWorldPosition = camera_data.PlayerPosition.xyz;
    vec3 SunDirection = camera_data.SunDirection.xyz;

    float Delta = chunk_data.delta_time; //May re-implement fading depending on performance.

    uint keep_index = lane * range.z;
    int is_lowest_lod = 1 - chunk_data.lowest_lod;

    for (uint target_index = range.x; target_index < range.y; target_index++) 
    {
        if (target_index >= chunk_data.total_foliage_instance_count){
            sKeep[keep_index] = 0.0;
            keep_index ++;
            continue;
        }

        uint current_index = target_index * 20u;
        
        vec3 instancePos = vec3(reference_instance.data[current_index + 3u], reference_instance.data[current_index + 7u], reference_instance.data[current_index + 11u]);
        float instanceScale = chunk_data.instance_frustum_bias * reference_instance.data[current_index + 5u];
        float blendValue = 0.0;
        float distance_lerp = length(playerWorldPosition - instancePos);
        // float frustum_edge = isInsideFrustum(instancePos, instanceScale);

        int inFrustum = min(isInsideFrustum(instancePos, instanceScale * 0.5), isInsideFrustum(instancePos + vec3(0.0, instanceScale, 0.0), instanceScale * 0.5)); // 0 = in frustum
        int inSun = isInSunLine(instancePos, playerWorldPosition, SunDirection, chunk_data.sun_instance_frustum_bias); // 0 = in sun line
        int inLoadRange = int(ceil(abs(clamp(distance_lerp, ViewDistance.x, ViewDistance.y) - distance_lerp))); //0 == in view range
        
        int load_behind = max(inSun, is_lowest_lod);

        if (inFrustum + inLoadRange == 0 || load_behind < inFrustum)
        {
            if (clamp(distance_lerp, ViewDistance.x + instanceScale, ViewDistance.y - instanceScale) == distance_lerp){
                blendValue = 1.0;
            }
            else{
                blendValue = clamp(reference_instance.data[current_index + 19u] + Delta, 0.0, 1.0);
            }
        } 
        else 
        {
            blendValue = clamp(reference_instance.data[current_index + 19u] - Delta, 0.0, 1.0);
        }

        sKeep[keep_index] = blendValue;
        sOrigins[keep_index] = target_index;
        keep_index ++;
    }

    barrier();
    memoryBarrierBuffer();
    
    if (lane == 0u) {
        uint totalKept = 0u;
        for (int search_keep_index = 0; search_keep_index < lane_count * range.z; search_keep_index++){
            if (sKeep[search_keep_index] > 0.0){
                sDestinations[search_keep_index] = totalKept;
                totalKept += 1u;
            }
        }
        sTotalKept = totalKept;
        // Only one atomic per batch for the entire workgroup
        sBase = (totalKept > 0u) ? atomicAdd(chunk_data.current_counter, totalKept) : 0u;
    }

    barrier();
    memoryBarrierBuffer();

    if (sTotalKept >= 0u){
        keep_index = lane * range.z;

        for (int outindex = range.x; outindex < range.y; outindex++) 
        {
            if (sKeep[keep_index] > 0.0){
                uint target_index = (sBase + sDestinations[keep_index]) * 20u;
                uint current_index = sOrigins[keep_index] * 20u;    
                
                for (int x = 0; x < 20; x++) {
                    output_instance.data[target_index + x] = reference_instance.data[current_index + x];
                }

                output_instance.data[target_index + 15u] = clamp(sKeep[keep_index] * 2.0, 0.0, 1.0);

                reference_instance.data[current_index + 19u] = sKeep[keep_index];
            }
            
            keep_index ++;
        }
    }
    barrier();
    memoryBarrierBuffer();

    if (lane == 0u) {
        uint prev = atomicAdd(chunk_data.done_counter, 1);
        if (prev + 1u == uint(chunk_data.workgroup_count)){
            // actually output the data
            for (int s = 0; s < chunk_data.mesh_surface_count; s++) {
                command_buffer.data[(s * 5) + 1] = int(chunk_data.current_counter);
            }

            // reset for next frame
            chunk_data.current_counter = 0u;
            chunk_data.done_counter    = 0u;
        }
    }
}
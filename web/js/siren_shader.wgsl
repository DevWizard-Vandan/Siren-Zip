// Spatio-Temporal SIREN & ACES Filmic Tone-Mapping WebGPU Compute Shader
// Evaluates continuous implicit neural coordinate field f_theta(x, y, t) -> (r, g, b) per pixel

struct Uniforms {
    viewport: vec4<f32>,        // [x_min, x_max, y_min, y_max]
    resolution: vec2<u32>,      // [canvas_width, canvas_height]
    time_and_params: vec4<f32>, // [t_local, omega_xy, omega_t, omega_0_hidden]
    model_meta: vec4<u32>,      // [hidden_features, hidden_layers, final_activation, tone_map_mode]
    extra_params: vec4<f32>,    // [zoom_factor, exposure, split_pos, is_split_mode]
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var out_texture: texture_storage_2d<rgba8unorm, write>;

// ACES Filmic Tone Mapping Curve
fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Reinhard Tone Mapping
fn reinhard(x: vec3<f32>) -> vec3<f32> {
    return clamp(x / (x + vec3<f32>(1.0)), vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let u_coord = global_id.x;
    let v_coord = global_id.y;
    let width = u.resolution.x;
    let height = u.resolution.y;

    if (u_coord >= width || v_coord >= height) {
        return;
    }

    // 1. Normalized continuous coordinates (x, y) in viewport bounds [x_min, x_max] x [y_min, y_max]
    let u_norm = (f32(u_coord) + 0.5) / f32(width);
    let v_norm = (f32(v_coord) + 0.5) / f32(height);

    let x = u.viewport.x + u_norm * (u.viewport.y - u.viewport.x);
    let y = u.viewport.z + v_norm * (u.viewport.w - u.viewport.z);
    let t = u.time_and_params.x;

    let omega_xy = u.time_and_params.y;
    let omega_t = u.time_and_params.z;
    let omega_0 = u.time_and_params.w;

    let H = u.model_meta.x; // hidden_features (e.g. 64, 128, 256, 384)
    let L = u.model_meta.y; // hidden_layers (e.g. 2, 4, 6)

    // Handle Split-Screen mode (discrete vs neural continuous)
    let is_split = u.extra_params.w > 0.5;
    let split_x = u.extra_params.z * f32(width);

    // Draw high-visibility split line
    if (is_split && abs(f32(u_coord) - split_x) < 2.0) {
        textureStore(out_texture, vec2<i32>(i32(u_coord), i32(v_coord)), vec4<f32>(0.0, 1.0, 0.4, 1.0));
        return;
    }

    // Ping-Pong activation buffers for hidden layers (supports up to H = 384)
    var h0: array<f32, 384>;
    var h1: array<f32, 384>;

    var offset: u32 = 0u;

    // --- 2. First Anisotropic Sine Layer (in_features = 3 -> (x*omega_xy, y*omega_xy, t*omega_t)) ---
    let scaled_in = vec3<f32>(x * omega_xy, y * omega_xy, t * omega_t);
    let w0_size = H * 3u;
    let b0_offset = w0_size;

    for (var i: u32 = 0u; i < H; i = i + 1u) {
        let w_idx = i * 3u;
        let w_x = weights[offset + w_idx + 0u];
        let w_y = weights[offset + w_idx + 1u];
        let w_t = weights[offset + w_idx + 2u];
        let b_val = weights[offset + b0_offset + i];

        let sum_val = w_x * scaled_in.x + w_y * scaled_in.y + w_t * scaled_in.z + b_val;
        h0[i] = sin(sum_val);
    }
    offset = offset + (H * 3u) + H;

    // --- 3. Hidden Sine Layers (l = 1 to L-1) ---
    var ping: u32 = 0u;
    for (var l: u32 = 0u; l < L - 1u; l = l + 1u) {
        let w_layer_size = H * H;
        let b_layer_offset = w_layer_size;

        if (ping == 0u) {
            for (var i: u32 = 0u; i < H; i = i + 1u) {
                var acc: f32 = 0.0;
                let row_idx = offset + i * H;
                for (var j: u32 = 0u; j < H; j = j + 1u) {
                    acc = acc + weights[row_idx + j] * h0[j];
                }
                let b_val = weights[offset + b_layer_offset + i];
                h1[i] = sin(omega_0 * (acc + b_val));
            }
            ping = 1u;
        } else {
            for (var i: u32 = 0u; i < H; i = i + 1u) {
                var acc: f32 = 0.0;
                let row_idx = offset + i * H;
                for (var j: u32 = 0u; j < H; j = j + 1u) {
                    acc = acc + weights[row_idx + j] * h1[j];
                }
                let b_val = weights[offset + b_layer_offset + i];
                h0[i] = sin(omega_0 * (acc + b_val));
            }
            ping = 0u;
        }
        offset = offset + (H * H) + H;
    }

    // --- 4. Final Linear Readout Layer (out_features = 3 for RGB) ---
    var rgb_out: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    let w_final_size = 3u * H;
    let b_final_offset = w_final_size;

    for (var c: u32 = 0u; c < 3u; c = c + 1u) {
        var acc: f32 = 0.0;
        let row_idx = offset + c * H;
        if (ping == 0u) {
            for (var j: u32 = 0u; j < H; j = j + 1u) {
                acc = acc + weights[row_idx + j] * h0[j];
            }
        } else {
            for (var j: u32 = 0u; j < H; j = j + 1u) {
                acc = acc + weights[row_idx + j] * h1[j];
            }
        }
        let b_val = weights[offset + b_final_offset + c];
        rgb_out[c] = acc + b_val;
    }

    // --- 5. Exposure & Final Tone Mapping ---
    let exposure = u.extra_params.y;
    var final_color = rgb_out * exposure;

    let tone_map_mode = u.model_meta.w; // 0=ACES, 1=Reinhard, 2=Linear Clamp
    if (tone_map_mode == 0u) {
        final_color = aces_filmic(final_color);
    } else if (tone_map_mode == 1u) {
        final_color = reinhard(final_color);
    } else {
        final_color = clamp(final_color, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    // Store pixel to output canvas texture
    textureStore(out_texture, vec2<i32>(i32(u_coord), i32(v_coord)), vec4<f32>(final_color, 1.0));
}

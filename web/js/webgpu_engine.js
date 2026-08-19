/**
 * WebGPU Compute Engine for Continuous Spatio-Temporal SIREN Evaluation.
 * Orchestrates GPU device initialization, dynamic WGSL compute pipelines, and 60 FPS canvas rendering.
 */

class WebGPUEngine {
    constructor() {
        this.canvas = null;
        this.context = null;
        this.device = null;
        this.adapter = null;
        this.adapterInfo = {};
        this.presentationFormat = null;

        this.computePipeline = null;
        this.renderPipeline = null;

        this.uniformBuffer = null;
        this.weightsBuffer = null;
        this.weightsBufferSize = 0;
        this.storageTexture = null;

        this.computeBindGroup = null;
        this.renderBindGroup = null;

        this.isReady = false;
        this.lastComputeTimeMs = 0;
    }

    /**
     * Initialize WebGPU Device, Shader Modules, and Compute Pipeline
     * @param {HTMLCanvasElement} canvas
     * @param {string} wgslCode
     */
    async init(canvas, wgslCode) {
        if (!navigator.gpu) {
            throw new Error("WebGPU is not supported on this browser. Please use Google Chrome 113+, Microsoft Edge, or enable chrome://flags/#enable-unsafe-webgpu.");
        }

        this.canvas = canvas;
        this.adapter = await navigator.gpu.requestAdapter({
            powerPreference: "high-performance",
        });

        if (!this.adapter) {
            throw new Error("No appropriate WebGPU graphics adapter found. Ensure your GPU drivers are up to date.");
        }

        try {
            this.adapterInfo = await this.adapter.requestAdapterInfo();
        } catch (e) {
            this.adapterInfo = { description: "High Performance GPU" };
        }

        this.device = await this.adapter.requestDevice();
        this.context = this.canvas.getContext("webgpu");
        this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: this.presentationFormat,
            alphaMode: "opaque",
        });

        // 1. Compile WGSL Compute Shader Module
        const shaderModule = this.device.createShaderModule({
            label: "SirenSpatioTemporalComputeShader",
            code: wgslCode,
        });

        // 2. Uniform Buffer (80 bytes aligned)
        this.uniformBuffer = this.device.createBuffer({
            label: "SirenUniforms",
            size: 80,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // 3. Create Compute Pipeline
        this.computePipeline = this.device.createComputePipeline({
            label: "SirenEvaluationPipeline",
            layout: "auto",
            compute: {
                module: shaderModule,
                entryPoint: "main",
            },
        });

        // 4. Create Fullscreen Quad Blit Render Pipeline
        this._initRenderPipeline();

        this._resizeStorageTexture(this.canvas.width, this.canvas.height);
        this.isReady = true;

        console.log(`[WebGPU] Initialized successfully on: ${this.getAdapterName()}`);
        return this.adapterInfo;
    }

    _initRenderPipeline() {
        const blitShader = `
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) uv: vec2<f32>,
            };

            @vertex
            fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
                var pos = array<vec2<f32>, 6>(
                    vec2<f32>(-1.0, -1.0),
                    vec2<f32>( 1.0, -1.0),
                    vec2<f32>(-1.0,  1.0),
                    vec2<f32>(-1.0,  1.0),
                    vec2<f32>( 1.0, -1.0),
                    vec2<f32>( 1.0,  1.0)
                );
                var uvs = array<vec2<f32>, 6>(
                    vec2<f32>(0.0, 1.0),
                    vec2<f32>(1.0, 1.0),
                    vec2<f32>(0.0, 0.0),
                    vec2<f32>(0.0, 0.0),
                    vec2<f32>(1.0, 1.0),
                    vec2<f32>(1.0, 0.0)
                );
                var out: VertexOutput;
                out.position = vec4<f32>(pos[vertex_index], 0.0, 1.0);
                out.uv = uvs[vertex_index];
                return out;
            }

            @group(0) @binding(0) var screen_tex: texture_2d<f32>;
            @group(0) @binding(1) var screen_sampler: sampler;

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                return textureSample(screen_tex, screen_sampler, in.uv);
            }
        `;

        const renderModule = this.device.createShaderModule({
            label: "BlitShaderModule",
            code: blitShader,
        });

        this.renderPipeline = this.device.createRenderPipeline({
            label: "BlitPipeline",
            layout: "auto",
            vertex: {
                module: renderModule,
                entryPoint: "vs_main",
            },
            fragment: {
                module: renderModule,
                entryPoint: "fs_main",
                targets: [{ format: this.presentationFormat }],
            },
            primitive: {
                topology: "triangle-list",
            },
        });

        this.sampler = this.device.createSampler({
            magFilter: "linear",
            minFilter: "linear",
        });
    }

    _resizeStorageTexture(width, height) {
        const w = Math.max(1, width);
        const h = Math.max(1, height);

        if (this.storageTexture && this.storageTexture.width === w && this.storageTexture.height === h) {
            return;
        }

        if (this.storageTexture) {
            this.storageTexture.destroy();
        }

        this.storageTexture = this.device.createTexture({
            label: "ComputeOutputTexture",
            size: [w, h, 1],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });

        this._rebuildBindGroups();
    }

    /**
     * Upload float32 model weights directly to WebGPU storage buffer
     * @param {Float32Array} weightsFloat32
     */
    updateWeights(weightsFloat32) {
        if (!weightsFloat32 || weightsFloat32.length === 0) return;

        const byteSize = weightsFloat32.byteLength;

        if (!this.weightsBuffer || this.weightsBufferSize < byteSize) {
            if (this.weightsBuffer) this.weightsBuffer.destroy();

            this.weightsBufferSize = Math.max(byteSize, 1024 * 1024); // at least 1MB
            this.weightsBuffer = this.device.createBuffer({
                label: "SirenWeightsStorageBuffer",
                size: this.weightsBufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            this._rebuildBindGroups();
        }

        this.device.queue.writeBuffer(this.weightsBuffer, 0, weightsFloat32);
    }

    _rebuildBindGroups() {
        if (!this.device || !this.computePipeline || !this.weightsBuffer || !this.storageTexture) {
            return;
        }

        this.computeBindGroup = this.device.createBindGroup({
            label: "SirenComputeBindGroup",
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.weightsBuffer } },
                { binding: 2, resource: this.storageTexture.createView() },
            ],
        });

        this.renderBindGroup = this.device.createBindGroup({
            label: "SirenRenderBindGroup",
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.storageTexture.createView() },
                { binding: 1, resource: this.sampler },
            ],
        });
    }

    /**
     * Execute high-throughput compute pass and present continuous field to canvas
     */
    render(params) {
        if (!this.isReady || !this.computeBindGroup || !this.renderBindGroup) return;

        const t0 = performance.now();
        const w = this.canvas.width;
        const h = this.canvas.height;

        this._resizeStorageTexture(w, h);

        // 1. Pack Uniforms Array Buffer (80 Bytes)
        const uniformData = new ArrayBuffer(80);
        const f32 = new Float32Array(uniformData);
        const u32 = new Uint32Array(uniformData);

        // Viewport: [x_min, x_max, y_min, y_max]
        const vp = params.viewport || { x_min: -1.0, x_max: 1.0, y_min: -1.0, y_max: 1.0 };
        f32[0] = vp.x_min;
        f32[1] = vp.x_max;
        f32[2] = vp.y_min;
        f32[3] = vp.y_max;

        // Resolution: [width, height]
        u32[4] = w;
        u32[5] = h;

        // Time & Params: [t_local, omega_xy, omega_t, omega_0_hidden]
        f32[6] = params.t_local !== undefined ? params.t_local : 0.0;
        f32[7] = params.omega_xy || 30.0;
        f32[8] = params.omega_t || 10.0;
        f32[9] = params.omega_0_hidden || 30.0;

        // Model Meta: [hidden_features, hidden_layers, final_activation, tone_map_mode]
        u32[10] = params.hidden_features || 256;
        u32[11] = params.hidden_layers || 5;
        u32[12] = 0; // clamp
        u32[13] = params.tone_map_mode === "reinhard" ? 1 : (params.tone_map_mode === "clamp" ? 2 : 0);

        // Extra Params: [zoom_factor, exposure, split_pos, is_split_mode]
        f32[14] = params.zoom_factor || 1.0;
        f32[15] = params.exposure || 1.0;
        f32[16] = params.split_pos || 0.5;
        f32[17] = params.is_split ? 1.0 : 0.0;

        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        // 2. Command Encoder: Compute Pass + Render Blit Pass
        const commandEncoder = this.device.createCommandEncoder();

        // Compute Pass
        const computePass = commandEncoder.beginComputePass({ label: "SirenComputePass" });
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16));
        computePass.end();

        // Render Blit Pass to HTML5 Canvas
        const currentTexture = this.context.getCurrentTexture();
        const renderPass = commandEncoder.beginRenderPass({
            label: "SirenCanvasRenderPass",
            colorAttachments: [{
                view: currentTexture.createView(),
                clearValue: { r: 0.05, g: 0.07, b: 0.09, a: 1.0 },
                loadOp: "clear",
                storeOp: "store",
            }],
        });
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);
        renderPass.draw(6);
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);

        this.lastComputeTimeMs = performance.now() - t0;
    }

    getAdapterName() {
        if (this.adapterInfo.description) return this.adapterInfo.description;
        if (this.adapterInfo.vendor) return `${this.adapterInfo.vendor} (${this.adapterInfo.architecture || 'WebGPU'})`;
        return "Dedicated WebGPU Graphics Device";
    }
}

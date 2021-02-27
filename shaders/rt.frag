#version 430 core

out vec4 FragColor;

// Computer vision style camera
struct Camera {
    mat4x3 transform;
    vec2 reso;
    float focal;
};

struct N3TreeSpec {
    int N;
    int data_dim;
    int sh_order;
    int n_coe;
    float ndc_width;
    float ndc_height;
    float ndc_focal;
    vec3 center;
    vec3 scale;
};

struct RenderOptions {
    // Epsilon added to each step
    float step_size;
    // If remaining light intensity/alpha < this amount stop marching
    float stop_thresh;
    // If sigma < this, skips
    float sigma_thresh;
    // Background brightness
    float background_brightness;
};

uniform Camera cam;
uniform RenderOptions opt;
uniform N3TreeSpec tree;

// The output image texture
layout(rgba32f, binding = 0) uniform image2D img_output;

// SSBO tree data
layout(std430, binding = 0) readonly restrict buffer TreeDataBuffer {
    float tree_data[];
};
layout(std430, binding = 1) readonly restrict buffer TreeChildBuffer {
    int tree_child[];
};

// **** N^3 TREE IMPLEMENTATION ****

// Tree query, returns
// (start index of leaf node in tree_data, leaf node scale 2^depth)
int query_single_from_root(inout vec3 xyz, out float cube_sz) {
    const float fN = tree.N;
    const int N3 = tree.N * tree.N * tree.N;
    xyz =  clamp(xyz, 0.f, 1.f - 1e-6f);
    int ptr = 0;
    int sub_ptr = 0;
    vec3 idx;
    for (cube_sz = 1; /*cube_sz < 11*/; ++cube_sz) {
        idx = floor(xyz * fN);

        // Find child offset
        sub_ptr = ptr + int(idx.x * (fN * fN) + idx.y * fN + idx.z);
        const int skip = tree_child[sub_ptr];
        xyz = xyz * fN - idx;

        // Add to output
        if (skip == 0) {
            break;
        }
        ptr += skip * N3;
    }
    cube_sz = pow(fN, cube_sz);
    return sub_ptr;
}


// **** CORE RAY TRACER IMPLEMENTATION ****

const float C0 = 0.28209479177387814;
const float C1 = 0.4886025119029199;
const float C2[] = {1.0925484305920792, -1.0925484305920792,
                    0.31539156525252005, -1.0925484305920792,
                    0.5462742152960396};

const float C3[] = {-0.5900435899266435, 2.890611442640554,
                    -0.4570457994644658, 0.3731763325901154,
                    -0.4570457994644658, 1.445305721320277,
                    -0.5900435899266435};

const float C4[] = {
    2.5033429417967046,  -1.7701307697799304, 0.9461746957575601,
    -0.6690465435572892, 0.10578554691520431, -0.6690465435572892,
    0.47308734787878004, -1.7701307697799304, 0.6258357354491761,
};

void precalc_sh(const int order, const vec3 dir, inout float out_mult[25]) {
    out_mult[0] = C0;
    const float x = dir[0], y = dir[1], z = dir[2];
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, yz = y * z, xz = x * z;
    switch (order) {
        case 4:
            out_mult[16] = C4[0] * xy * (xx - yy);
            out_mult[17] = C4[1] * yz * (3 * xx - yy);
            out_mult[18] = C4[2] * xy * (7 * zz - 1.f);
            out_mult[19] = C4[3] * yz * (7 * zz - 3.f);
            out_mult[20] = C4[4] * (zz * (35 * zz - 30) + 3);
            out_mult[21] = C4[5] * xz * (7 * zz - 3);
            out_mult[22] = C4[6] * (xx - yy) * (7 * zz - 1.f);
            out_mult[23] = C4[7] * xz * (xx - 3 * yy);
            out_mult[24] = C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
        case 3:
            out_mult[9] = C3[0] * y * (3 * xx - yy);
            out_mult[10] = C3[1] * xy * z;
            out_mult[11] = C3[2] * y * (4 * zz - xx - yy);
            out_mult[12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
            out_mult[13] = C3[4] * x * (4 * zz - xx - yy);
            out_mult[14] = C3[5] * z * (xx - yy);
            out_mult[15] = C3[6] * x * (xx - 3 * yy);
        case 2:
            out_mult[4] = C2[0] * xy;
            out_mult[5] = C2[1] * yz;
            out_mult[6] = C2[2] * (2.0 * zz - xx - yy);
            out_mult[7] = C2[3] * xz;
            out_mult[8] = C2[4] * (xx - yy);
        case 1:
            out_mult[1] = -C1 * y;
            out_mult[2] = C1 * z;
            out_mult[3] = -C1 * x;
    }
}

void dda_unit(vec3 cen, vec3 _invdir, out float tmin, out float tmax) {
    float t1, t2;
    tmin = 0.0f;
    tmax = 1e9f;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        t1 = - cen[i] * _invdir[i];
        t2 = t1 +  _invdir[i];
        tmin = max(tmin, min(t1, t2));
        tmax = min(tmax, max(t1, t2));
    }
}

float _get_delta_scale(vec3 scaling, inout vec3 dir) {
    dir *= scaling;
    float delta_scale = 1.0 / length(dir);
    dir *= delta_scale;
    return delta_scale;
}

vec3 trace_ray(vec3 dir, vec3 vdir, vec3 cen) {
    float delta_scale = _get_delta_scale(tree.scale, dir);
    vec3 output_color;
    vec3 invdir = 1.f / (dir + 1e-9);
    float tmin, tmax;
    dda_unit(cen, invdir, tmin, tmax);

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        output_color = vec3(opt.background_brightness);
    } else {
        output_color = vec3(.0f);
        float sh_mult[25];
        if (tree.sh_order >= 0) {
            precalc_sh(tree.sh_order, vdir, sh_mult);
        }

        float light_intensity = 1.f;
        float t = tmin;
        float cube_sz;
        // int n_steps = 0;
        while (t < tmax) {
            // ++n_steps;
            vec3 pos = cen + t * dir;
            float cube_sz;
            int doffset = query_single_from_root(pos, cube_sz) * tree.data_dim;

            float subcube_tmin, subcube_tmax;
            dda_unit(pos, invdir, subcube_tmin, subcube_tmax);

            float t_subcube = (subcube_tmax - subcube_tmin) / float(cube_sz);

            const float delta_t = t_subcube + opt.step_size;
            if (tree_data[doffset + tree.data_dim - 1] > opt.sigma_thresh) {
                const float att = min(exp(-delta_t * delta_scale * tree_data[
                            doffset + tree.data_dim - 1]), 1.f);
                const float weight = light_intensity * (1.f - att);

                if (tree.sh_order >= 0) {
                    int off = 0;
                    for (int t = 0; t < 3; ++ t) {
                        float tmp = sh_mult[0] * tree_data[doffset + off] +
                            sh_mult[1] * tree_data[doffset + off + 1] +
                            sh_mult[2] * tree_data[doffset + off + 2];
                        for (int i = 3; i < tree.n_coe; ++i) {
                            tmp += sh_mult[i] * tree_data[doffset + off + i];
                        }
                        output_color[t] += weight / (1.f + exp(-tmp));
                        off += tree.n_coe;
                    }
                } else {
                    for (int j = 0; j < 3; ++j) {
                        output_color[j] += tree_data[doffset + j] * weight;
                    }
                }

                light_intensity *= att;
                if (light_intensity < opt.stop_thresh) {
                    // Almost full opacity, stop
                    output_color *= 1.f / (1.f - light_intensity);
                    light_intensity = 0.f;
                    break;
                }
            }
            t += delta_t;
        }
        output_color += light_intensity * opt.background_brightness;
        // output_color = vec3(min(float(n_steps) / 255, 1.0));
        // output_color = vec3(min(0.5 * (tbounds.y - tbounds.x), 1.0f));
        return output_color;
    }
    return output_color;
}

// **** NDC ****
void world2ndc(inout vec3 dir, inout vec3 cen, float near = 1.f) {
    float t = -(near + cen[2]) / dir[2];
    cen += t * dir;

    dir[0] = -((2 * tree.ndc_focal) / tree.ndc_width) * (dir[0] / dir[2] - cen[0] / cen[2]);
    dir[1] = -((2 * tree.ndc_focal) / tree.ndc_height) * (dir[1] / dir[2] - cen[1] / cen[2]);
    dir[2] = -2 * near / cen[2];

    cen[0] = -((2 * tree.ndc_focal) / tree.ndc_width) * (cen[0] / cen[2]);
    cen[1] = -((2 * tree.ndc_focal) / tree.ndc_height) * (cen[1] / cen[2]);
    cen[2] = 1 + 2 * near / cen[2];

    dir = normalize(dir);
}

void main()
{
    vec2 xy = vec2(gl_FragCoord);
    xy.y = cam.reso.y - xy.y;
    xy = (xy - 0.5 * (cam.reso + 1)) / cam.focal * vec2(1, -1);
    vec3 dir = normalize(vec3(xy, -1.0));
    dir = normalize(mat3(cam.transform) * dir);
    vec3 cen = cam.transform[3];

    vec3 rgb;
    vec3 vdir = dir;
    if (tree.ndc_width > 0.f) {
        world2ndc(dir, cen);
    }
    cen = tree.center + cen * tree.scale;
    rgb = trace_ray(dir, vdir, cen);
    FragColor = vec4(rgb, 1.0);
}

#version 330 core

precision lowp float;

// The output color
out vec4 FragColor;

// Computer vision style camera
struct Camera {
    mat4x3 transform;
    vec2 reso;
    float focal;
};

// Store tree data
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

// Store render options
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

uniform int tbo_size_limit;
uniform isamplerBuffer tree_child_tex;
uniform samplerBuffer tree_data_tex[8];


// **** N^3 TREE IMPLEMENTATION ****

// Tree query, returns
// (start index of leaf node in tree_data, leaf node scale 2^depth)
int query_single_from_root(inout vec3 xyz, out float cube_sz) {
    float fN = tree.N;
    int N3 = tree.N * tree.N * tree.N;
    xyz =  clamp(xyz, 0.f, 1.f - 1e-6f);
    int ptr = 0;
    int sub_ptr = 0;
    vec3 idx;
    for (cube_sz = 1; /*cube_sz < 11*/; ++cube_sz) {
        idx = floor(xyz * fN);

        // Find child offset
        sub_ptr = ptr + int(idx.x * (fN * fN) + idx.y * fN + idx.z);
        int skip = texelFetch(tree_child_tex, sub_ptr).r;
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
void precalc_sh(const int order, const vec3 dir, inout float out_mult[25]) {
    out_mult[0] = 0.28209479177387814;
    const float x = dir[0], y = dir[1], z = dir[2];
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, yz = y * z, xz = x * z;
    switch (order) {
        case 4:
            out_mult[16] = 2.5033429417967046 * xy * (xx - yy);
            out_mult[17] = -1.7701307697799304 * yz * (3 * xx - yy);
            out_mult[18] = 0.9461746957575601 * xy * (7 * zz - 1.f);
            out_mult[19] = -0.6690465435572892 * yz * (7 * zz - 3.f);
            out_mult[20] = 0.10578554691520431 * (zz * (35 * zz - 30) + 3);
            out_mult[21] = -0.6690465435572892 * xz * (7 * zz - 3);
            out_mult[22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1.f);
            out_mult[23] = -1.7701307697799304 * xz * (xx - 3 * yy);
            out_mult[24] = 0.6258357354491761 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
        case 3:
            out_mult[9] = -0.5900435899266435 * y * (3 * xx - yy);
            out_mult[10] = 2.890611442640554 * xy * z;
            out_mult[11] = -0.4570457994644658 * y * (4 * zz - xx - yy);
            out_mult[12] = 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy);
            out_mult[13] = -0.4570457994644658 * x * (4 * zz - xx - yy);
            out_mult[14] = 1.445305721320277 * z * (xx - yy);
            out_mult[15] = -0.5900435899266435 * x * (xx - 3 * yy);
        case 2:
            out_mult[4] = 1.0925484305920792 * xy;
            out_mult[5] = -1.0925484305920792 * yz;
            out_mult[6] = 0.31539156525252005 * (2.0 * zz - xx - yy);
            out_mult[7] = -1.0925484305920792 * xz;
            out_mult[8] = 0.5462742152960396 * (xx - yy);
        case 1:
            out_mult[1] = -0.4886025119029199 * y;
            out_mult[2] = 0.4886025119029199 * z;
            out_mult[3] = -0.4886025119029199 * x;
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

float index_tree_data(int i) {
    return texelFetch(tree_data_tex[i / tbo_size_limit], i % tbo_size_limit).r;
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

            float delta_t = t_subcube + opt.step_size;
            if (index_tree_data(doffset + tree.data_dim - 1) > opt.sigma_thresh) {
                float att = min(exp(-delta_t * delta_scale * index_tree_data(
                            doffset + tree.data_dim - 1)), 1.f);
                float weight = light_intensity * (1.f - att);

                if (tree.sh_order >= 0) {
                    int off = 0;
                    for (int t = 0; t < 3; ++ t) {
                        float tmp = sh_mult[0] * index_tree_data(doffset + off) +
                            sh_mult[1] * index_tree_data(doffset + off + 1) +
                            sh_mult[2] * index_tree_data(doffset + off + 2);
                        for (int i = 3; i < tree.n_coe; ++i) {
                            tmp += sh_mult[i] * index_tree_data(doffset + off + i);
                        }
                        output_color[t] += weight / (1.f + exp(-tmp));
                        off += tree.n_coe;
                    }
                } else {
                    for (int j = 0; j < 3; ++j) {
                        output_color[j] += index_tree_data(doffset + j) * weight;
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
    vec2 xy = (vec2(gl_FragCoord) - 0.5 * cam.reso + vec2(-0.5, 0.5)) / cam.focal;
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

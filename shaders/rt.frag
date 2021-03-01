#version 300 es

precision highp float;
precision highp int;

#define VOLREND_GLOBAL_BASIS_MAX 25

#define FORMAT_RGBA 0
#define FORMAT_SH 1
#define FORMAT_SG 2
#define FORMAT_ASG 3

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
    int format;
    int basis_dim;
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

uniform int tree_child_dim;
uniform highp isampler2D tree_child_tex;
uniform int tree_data_dim;
uniform highp sampler2D tree_data_tex;
uniform highp sampler2D tree_extra_tex;

// Hacky ways to store octree in 2 textures
float index_tree_data(int i) {
    int y = i / tree_data_dim;
    int x = i % tree_data_dim;
    return texelFetch(tree_data_tex, ivec2(x, y), 0).r;
}

int index_tree_child(int i) {
    int y = i / tree_child_dim;
    int x = i % tree_child_dim;
    return texelFetch(tree_child_tex, ivec2(x, y), 0).r;
}

float index_extra(int i, int j) {
    return texelFetch(tree_extra_tex, ivec2(j, i), 0).r;
}


// **** N^3 TREE IMPLEMENTATION ****

// Tree query, returns
// (start index of leaf node in tree_data, leaf node scale 2^depth)
int query_single_from_root(inout vec3 xyz, out float cube_sz) {
    float fN = float(tree.N);
    int N3 = tree.N * tree.N * tree.N;
    xyz =  clamp(xyz, 0.f, 1.f - 1e-6f);
    int ptr = 0;
    int sub_ptr = 0;
    vec3 idx;
    for (cube_sz = 1.f; /*cube_sz < 11*/; ++cube_sz) {
        idx = floor(xyz * fN);

        // Find child offset
        sub_ptr = ptr + int(idx.x * (fN * fN) + idx.y * fN + idx.z);
        int skip = index_tree_child(sub_ptr);
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
void maybe_precalc_sh(const vec3 dir, inout float outb[VOLREND_GLOBAL_BASIS_MAX]) {
    switch(tree.format) {
        case FORMAT_ASG:
            {
                for (int i = 0; i < tree.basis_dim; ++i) {
                    float lambda_x = index_extra(i, 0);
                    float lambda_y = index_extra(i, 1);
                    vec3 mu_x = vec3(index_extra(i, 2), index_extra(i, 3), index_extra(i, 4));
                    vec3 mu_y = vec3(index_extra(i, 5), index_extra(i, 6), index_extra(i, 7));
                    vec3 mu_z = vec3(index_extra(i, 8), index_extra(i, 9), index_extra(i, 10));
                    float S = dot(dir, mu_z);
                    float dot_x = dot(dir, mu_x);
                    float dot_y = dot(dir, mu_y);
                    outb[i] = S * exp(-lambda_x * dot_x * dot_x
                                      -lambda_y * dot_y * dot_y) / float(tree.basis_dim);
                }
            }
            break;
        case FORMAT_SG:
            {
                for (int i = 0; i < tree.basis_dim; ++i) {
                    vec3 mu = vec3(index_extra(i, 1), index_extra(i, 2), index_extra(i, 3));
                    outb[i] = exp(index_extra(i, 0) * (dot(dir, mu) - 1.f)) /
                                  float(tree.basis_dim);
                }
            }
            break;
        case FORMAT_SH:
            {
                outb[0] = 0.28209479177387814;
                float x = dir[0], y = dir[1], z = dir[2];
                float xx = x * x, yy = y * y, zz = z * z;
                float xy = x * y, yz = y * z, xz = x * z;
                switch (tree.basis_dim) {
                    case 25:
                        outb[16] = 2.5033429417967046 * xy * (xx - yy);
                        outb[17] = -1.7701307697799304 * yz * (3.f * xx - yy);
                        outb[18] = 0.9461746957575601 * xy * (7.f * zz - 1.f);
                        outb[19] = -0.6690465435572892 * yz * (7.f * zz - 3.f);
                        outb[20] = 0.10578554691520431 * (zz * (35.f * zz - 30.f) + 3.f);
                        outb[21] = -0.6690465435572892 * xz * (7.f * zz - 3.f);
                        outb[22] = 0.47308734787878004 * (xx - yy) * (7.f * zz - 1.f);
                        outb[23] = -1.7701307697799304 * xz * (xx - 3.f * yy);
                        outb[24] = 0.6258357354491761 * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy));
                    case 16:
                        outb[9] = -0.5900435899266435 * y * (3.f * xx - yy);
                        outb[10] = 2.890611442640554 * xy * z;
                        outb[11] = -0.4570457994644658 * y * (4.f * zz - xx - yy);
                        outb[12] = 0.3731763325901154 * z * (2.f * zz - 3.f * xx - 3.f * yy);
                        outb[13] = -0.4570457994644658 * x * (4.f * zz - xx - yy);
                        outb[14] = 1.445305721320277 * z * (xx - yy);
                        outb[15] = -0.5900435899266435 * x * (xx - 3.f * yy);
                    case 9:
                        outb[4] = 1.0925484305920792 * xy;
                        outb[5] = -1.0925484305920792 * yz;
                        outb[6] = 0.31539156525252005 * (2.f * zz - xx - yy);
                        outb[7] = -1.0925484305920792 * xz;
                        outb[8] = 0.5462742152960396 * (xx - yy);
                    case 4:
                        outb[1] = -0.4886025119029199 * y;
                        outb[2] = 0.4886025119029199 * z;
                        outb[3] = -0.4886025119029199 * x;
                }
        }
    }
}

void dda_unit(vec3 cen, vec3 _invdir, out float tmin, out float tmax) {
    float t1, t2;
    tmin = 0.0f;
    tmax = 1e9f;
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

    if (tmax < 0.f || tmin > tmax) {
        // Ray doesn't hit box
        output_color = vec3(opt.background_brightness);
    } else {
        output_color = vec3(.0f);
        float basis_fn[VOLREND_GLOBAL_BASIS_MAX];
        maybe_precalc_sh(vdir, basis_fn);

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

                if (tree.format != FORMAT_RGBA) {
                    int off = 0;
                    for (int t = 0; t < 3; ++ t) {
                        float tmp = basis_fn[0] * index_tree_data(doffset + off) +
                            basis_fn[1] * index_tree_data(doffset + off + 1) +
                            basis_fn[2] * index_tree_data(doffset + off + 2);
                        for (int i = 3; i < tree.basis_dim; ++i) {
                            tmp += basis_fn[i] * index_tree_data(doffset + off + i);
                        }
                        output_color[t] += weight / (1.f + exp(-tmp));
                        off += tree.basis_dim;
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
void world2ndc(inout vec3 dir, inout vec3 cen, float near) {
    float t = -(near + cen[2]) / dir[2];
    cen += t * dir;

    dir[0] = -((2.f * tree.ndc_focal) / tree.ndc_width) * (dir[0] / dir[2] - cen[0] / cen[2]);
    dir[1] = -((2.f * tree.ndc_focal) / tree.ndc_height) * (dir[1] / dir[2] - cen[1] / cen[2]);
    dir[2] = -2.f * near / cen[2];

    cen[0] = -((2.f * tree.ndc_focal) / tree.ndc_width) * (cen[0] / cen[2]);
    cen[1] = -((2.f * tree.ndc_focal) / tree.ndc_height) * (cen[1] / cen[2]);
    cen[2] = 1.f + 2.f * near / cen[2];

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
        world2ndc(dir, cen, 1.f);
    }
    cen = tree.center + cen * tree.scale;
    rgb = trace_ray(dir, vdir, cen);
    rgb = clamp(rgb, 0.0, 1.0);
    FragColor = vec4(rgb, 1.0);
}

// The PlenOctree shader, kind of crappy performance

// VERTEX SHADER
#if defined(VERTEX)

in vec3 aPos;

void main()
{
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
}

// FRAGMENT SHADER
#elif defined(FRAGMENT)

precision highp float;
precision highp int;

#define VOLREND_GLOBAL_BASIS_MAX 16

#define FORMAT_RGBA 0
#define FORMAT_SH 1
#define FORMAT_SG 2
#define FORMAT_ASG 3

// The output color
layout(location = 0) out vec4 FragColor;
layout(location = 1) out float Depth;

// Computer vision style camera
struct Camera {
    mat4x3 transform;
    vec2 reso;
    vec2 focal;
};

// Store tree data
struct N3TreeSpec {
    int data_dim;
    int data_dim_rgba;
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

    // Rendering bounding box (relative to outer tree bounding box [0, 1])
    // [minx, miny, minz, maxx, maxy, maxz]
    float render_bbox[6];
    // Range of basis functions to use
    int basis_minmax[2];
    // Rotation applied to viewdirs for all rays
    vec3 rot_dirs;
};

uniform Camera cam;
uniform RenderOptions opt;
uniform N3TreeSpec tree;

uniform int tree_child_stride;
uniform highp isampler2D tree_child_tex;
uniform int tree_data_stride;
uniform mediump sampler2D tree_data_tex;
// uniform highp sampler2D tree_extra_tex;

// Mesh rendering compositing
uniform mediump sampler2D mesh_depth_tex;
uniform mediump sampler2D mesh_color_tex;

// Hacky ways to store octree in 2 textures
vec4 get_tree_data(int y, int x) {
    return texelFetch(tree_data_tex, ivec2(x, y), 0);
}
int index_tree_child(int i) {
    int y = i / tree_child_stride;
    int x = i % tree_child_stride;
    return texelFetch(tree_child_tex, ivec2(x, y), 0).r;
}


// **** N^3 TREE IMPLEMENTATION ****

// Tree query, returns
// (start index of leaf node in tree_data, leaf node scale 2^depth)
int query_single_from_root(inout vec3 xyz, out float cube_sz) {
    xyz =  clamp(xyz, 0.f, 1.f - 1e-6f);
    int ptr = 0;
    int sub_ptr = 0;
    vec3 idx;
    for (cube_sz = 1.f; /*cube_sz < 11*/; ++cube_sz) {
        idx = floor(xyz * 2.0);

        // Find child offset
        sub_ptr = ptr + int(idx.x * 4.0 + idx.y * 2.0 + idx.z);
        int skip = index_tree_child(sub_ptr);
        xyz = xyz * 2.0 - idx;

        // Add to output
        if (skip == 0) {
            break;
        }
        ptr += skip * 8;
    }
    cube_sz = pow(2.0, cube_sz);
    return sub_ptr;
}


void rodrigues(vec3 aa, inout vec3 dir) {
    float angle = length(aa);
    if (angle < 1e-6) return;
    vec3 k = aa / angle;
    float cos_angle = cos(angle);
    float sin_angle = sin(angle);
    vec3 cp = cross(k, dir);
    float dot = dot(k, dir);
    dir = dir * cos_angle + cp * sin_angle + k * dot * (1.0 - cos_angle);
}


// **** CORE RAY TRACER IMPLEMENTATION ****
void maybe_precalc_basis(const vec3 dir, inout float outb[VOLREND_GLOBAL_BASIS_MAX]) {
    {
        outb[0] = 0.28209479177387814;
        float x = dir[0], y = dir[1], z = dir[2];
        float xx = x * x, yy = y * y, zz = z * z;
        float xy = x * y, yz = y * z, xz = x * z;
        switch (tree.basis_dim) {
            // case 25:
            //     outb[16] = 2.5033429417967046 * xy * (xx - yy);
            //     outb[17] = -1.7701307697799304 * yz * (3.f * xx - yy);
            //     outb[18] = 0.9461746957575601 * xy * (7.f * zz - 1.f);
            //     outb[19] = -0.6690465435572892 * yz * (7.f * zz - 3.f);
            //     outb[20] = 0.10578554691520431 * (zz * (35.f * zz - 30.f) + 3.f);
            //     outb[21] = -0.6690465435572892 * xz * (7.f * zz - 3.f);
            //     outb[22] = 0.47308734787878004 * (xx - yy) * (7.f * zz - 1.f);
            //     outb[23] = -1.7701307697799304 * xz * (xx - 3.f * yy);
            //     outb[24] = 0.6258357354491761 * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy));
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

void dda_world(vec3 cen, vec3 _invdir, out float tmin, out float tmax) {
    float t1, t2;
    tmin = 0.0f;
    tmax = 1e9f;
    for (int i = 0; i < 3; ++i) {
        t1 = (opt.render_bbox[i] - cen[i]) * _invdir[i];
        t2 = (opt.render_bbox[i + 3] - cen[i]) * _invdir[i];
        tmin = max(tmin, min(t1, t2));
        tmax = min(tmax, max(t1, t2));
    }
}

void dda_unit(vec3 cen, vec3 _invdir, out float tmax) {
    float t1, t2;
    tmax = 1e9f;
    for (int i = 0; i < 3; ++i) {
        t1 = - cen[i] * _invdir[i];
        t2 = t1 +  _invdir[i];
        tmax = min(tmax, max(t1, t2));
    }
}

float _get_delta_scale(vec3 scaling, inout vec3 dir) {
    dir *= scaling;
    float delta_scale = 1.0 / length(dir);
    dir *= delta_scale;
    return delta_scale;
}

vec4 trace_ray(vec3 dir, vec3 vdir, vec3 cen, inout float tmax_bg, vec4 bg_color) {
    float delta_scale = _get_delta_scale(tree.scale, dir);
    vec4 output_color;
    vec3 invdir = 1.f / (dir + 1e-9);
    float tmin, tmax;
    dda_world(cen, invdir, tmin, tmax);
    tmax = min(tmax, tmax_bg / delta_scale);

    if (tmax < 0.f || tmin > tmax || tree_data_stride == 0) {
        // Ray doesn't hit box or tree not loaded
        output_color = bg_color;
    } else {
        output_color = vec4(0.0, 0.0, 0.0, 0.0);
        float basis_fn[VOLREND_GLOBAL_BASIS_MAX];
        maybe_precalc_basis(vdir, basis_fn);
        for (int i = 0; i < opt.basis_minmax[0]; ++i) {
            basis_fn[i] = 0.f;
        }
        for (int i = opt.basis_minmax[1] + 1; i < VOLREND_GLOBAL_BASIS_MAX; ++i) {
            basis_fn[i] = 0.f;
        }

        float light_intensity = 1.f;
        float t = tmin;
        float depth = -1.f;
        float cube_sz;
        // int n_steps = 0;
        while (t < tmax) {
            // ++n_steps;
            vec3 pos = cen + t * dir;
            float cube_sz;
            int doffset = query_single_from_root(pos, cube_sz) * tree.data_dim_rgba;

            float subcube_tmax;
            dda_unit(pos, invdir, subcube_tmax);

            float t_subcube = subcube_tmax / float(cube_sz);
            int tree_y = doffset / tree_data_stride;
            int tree_x = doffset % tree_data_stride;

            float delta_t = t_subcube + opt.step_size;
            vec4 data = get_tree_data(tree_y, tree_x);
            if (data.r > opt.sigma_thresh) {
                if (data.r > 10.0) {
                    // Approximate multiple plenoctree compositing using median depth
                    depth = t;
                }
                float att = min(exp(-delta_t * delta_scale * data.r), 1.f);
                float weight = light_intensity * (1.f - att);

                if (tree.format != FORMAT_RGBA) {
                    float tmp;
                    if (tree.basis_dim == 1) {
                        output_color.rgb += (weight * basis_fn[0]) * data.gba;
                    } else if (tree.basis_dim == 4) {
                        tmp = basis_fn[0] * data.g
                              + basis_fn[1] * data.b
                              + basis_fn[2] * data.a;
                        data = get_tree_data(tree_y, ++tree_x);
                        tmp += basis_fn[3] * data.r;
                        output_color.r += weight / (1.0 + exp(-tmp));

                        tmp = basis_fn[0] * data.g
                              + basis_fn[1] * data.b
                              + basis_fn[2] * data.a;
                        data = get_tree_data(tree_y, ++tree_x);
                        tmp += basis_fn[3] * data.r;
                        output_color.g += weight / (1.0 + exp(-tmp));

                        tmp = basis_fn[0] * data.g
                              + basis_fn[1] * data.b
                              + basis_fn[2] * data.a;
                        data = get_tree_data(tree_y, ++tree_x);
                        tmp += basis_fn[3] * data.r;
                        output_color.b += weight / (1.0 + exp(-tmp));
                    } else if (tree.basis_dim == 9) {
                        tmp = basis_fn[0] * data.g
                              + basis_fn[1] * data.b
                              + basis_fn[2] * data.a;
                        data = get_tree_data(tree_y, ++tree_x);
                        tmp += basis_fn[3] * data.r
                              + basis_fn[4] * data.g
                              + basis_fn[5] * data.b
                              + basis_fn[6] * data.a;
                        data = get_tree_data(tree_y, ++tree_x);
                        tmp += basis_fn[7] * data.r
                              + basis_fn[8] * data.g;
                        output_color.r += weight / (1.0 + exp(-tmp));

                        tmp = basis_fn[0] * data.b
                            + basis_fn[1] * data.a;
                        data = get_tree_data(tree_y, ++tree_x);
                        tmp += basis_fn[2] * data.r
                             + basis_fn[3] * data.g
                             + basis_fn[4] * data.b
                             + basis_fn[5] * data.a;
                        data = get_tree_data(tree_y, ++tree_x);
                        tmp += basis_fn[6] * data.r
                             + basis_fn[7] * data.g
                             + basis_fn[8] * data.b;
                        output_color.g += weight / (1.0 + exp(-tmp));

                        tmp = basis_fn[0] * data.a;
                        data = get_tree_data(tree_y, ++tree_x);
                        tmp += basis_fn[1] * data.r
                             + basis_fn[2] * data.g
                             + basis_fn[3] * data.b
                             + basis_fn[4] * data.a;
                        data = get_tree_data(tree_y, ++tree_x);
                        tmp += basis_fn[5] * data.r
                             + basis_fn[6] * data.g
                             + basis_fn[7] * data.b
                             + basis_fn[8] * data.a;
                        output_color.b += weight / (1.0 + exp(-tmp));
                    } else if (tree.basis_dim == 16) {
                        tmp = basis_fn[0] * data.g
                            + basis_fn[1] * data.b
                            + basis_fn[2] * data.a;
                        data = get_tree_data(tree_y, ++tree_x);
                        tmp += basis_fn[3] * data.r
                            + basis_fn[4] * data.g
                            + basis_fn[5] * data.b
                            + basis_fn[6] * data.a;
                        data = get_tree_data(tree_y, ++tree_x);
                        tmp += basis_fn[7] * data.r
                            + basis_fn[8] * data.g
                            + basis_fn[9] * data.b
                            + basis_fn[10] * data.a;
                        data = get_tree_data(tree_y, ++tree_x);
                        tmp += basis_fn[11] * data.r
                            + basis_fn[12] * data.g
                            + basis_fn[13] * data.b
                            + basis_fn[14] * data.a;
                        data = get_tree_data(tree_y, ++tree_x);
                        tmp += basis_fn[11] * data.r;
                        output_color.r += weight / (1.0 + exp(-tmp));

                        tmp = basis_fn[0] * data.g
                            + basis_fn[1] * data.b
                            + basis_fn[2] * data.a;
                        data = get_tree_data(tree_y, ++tree_x);
                        tmp += basis_fn[3] * data.r
                            + basis_fn[4] * data.g
                            + basis_fn[5] * data.b
                            + basis_fn[6] * data.a;
                        data = get_tree_data(tree_y, ++tree_x);
                        tmp += basis_fn[7] * data.r
                            + basis_fn[8] * data.g
                            + basis_fn[9] * data.b
                            + basis_fn[10] * data.a;
                        data = get_tree_data(tree_y, ++tree_x);
                        tmp += basis_fn[11] * data.r
                            + basis_fn[12] * data.g
                            + basis_fn[13] * data.b
                            + basis_fn[14] * data.a;
                        data = get_tree_data(tree_y, ++tree_x);
                        tmp += basis_fn[11] * data.r;
                        output_color.g += weight / (1.0 + exp(-tmp));

                        tmp = basis_fn[0] * data.g
                            + basis_fn[1] * data.b
                            + basis_fn[2] * data.a;
                        data = get_tree_data(tree_y, ++tree_x);
                        tmp += basis_fn[3] * data.r
                            + basis_fn[4] * data.g
                            + basis_fn[5] * data.b
                            + basis_fn[6] * data.a;
                        data = get_tree_data(tree_y, ++tree_x);
                        tmp += basis_fn[7] * data.r
                            + basis_fn[8] * data.g
                            + basis_fn[9] * data.b
                            + basis_fn[10] * data.a;
                        data = get_tree_data(tree_y, ++tree_x);
                        tmp += basis_fn[11] * data.r
                            + basis_fn[12] * data.g
                            + basis_fn[13] * data.b
                            + basis_fn[14] * data.a;
                        data = get_tree_data(tree_y, ++tree_x);
                        tmp += basis_fn[11] * data.r;
                        output_color.b += weight / (1.0 + exp(-tmp));
                    }
                } else {
                    output_color.rgb += weight * data.gba;
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

        output_color.a = 1.0 - light_intensity;
        if (output_color.a > 1e-4 && depth >= 0.f) {
            tmax_bg = depth * delta_scale;
        }
        output_color += light_intensity * bg_color;
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

    vec3 vdir = dir;
    if (tree.ndc_width > 0.f) {
        world2ndc(dir, cen, 1.f);
    }
    cen = tree.center + cen * tree.scale;
    rodrigues(opt.rot_dirs, vdir);

    // Get depth of drawn meshes and other plenoctrees
    ivec2 screen_pt = ivec2(gl_FragCoord.x, gl_FragCoord.y);
    float tmax_bg = texelFetch(mesh_depth_tex, screen_pt, 0).r;
    vec4 mesh_color = texelFetch(mesh_color_tex, screen_pt, 0);
    vec4 bg_color = mesh_color;

    vec4 out_color = trace_ray(dir, vdir, cen, tmax_bg, bg_color);
    out_color = clamp(out_color, 0.0, 1.0);
    FragColor = out_color;
    Depth = tmax_bg;
}
#endif

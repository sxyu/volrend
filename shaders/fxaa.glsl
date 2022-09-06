/**
Basic FXAA implementation based on the code on geeks3d.com with the
modification that the texture2DLod stuff was removed since it's
unsupported by WebGL.

Taken from https://github.com/mattdesl/glsl-fxaa
(MIT License)

--

Original from:
https://github.com/mitsuhiko/webgl-meincraft

Copyright (c) 2011 by Armin Ronacher.

Some rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    * The names of the contributors may not be used to endorse or
      promote products derived from this software without specific
      prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// VERTEX SHADER
#if defined(VERTEX)
#if __VERSION__ >= 130
#define COMPAT_ATTRIBUTE in
#else
#define COMPAT_ATTRIBUTE attribute
#endif
precision highp float;
precision highp int;
COMPAT_ATTRIBUTE vec3 aPos;

void main()
{
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
}


// FRAGMENT SHADER
#elif defined(FRAGMENT)

#if __VERSION__ >= 130
#define COMPAT_TEXTURE texture
#else
#define COMPAT_TEXTURE texture2D
#endif


precision mediump float;

uniform vec2 resolution;
uniform sampler2D tex_input;

#ifndef FXAA_REDUCE_MIN
    #define FXAA_REDUCE_MIN   (1.0/ 128.0)
#endif
#ifndef FXAA_REDUCE_MUL
    #define FXAA_REDUCE_MUL   (1.0 / 8.0)
#endif
#ifndef FXAA_SPAN_MAX
    #define FXAA_SPAN_MAX     8.0
#endif

#if __VERSION__ >= 130
out vec4 FragColor;
#endif

//optimized version for mobile, where dependent
//texture reads can be a bottleneck
vec3 rgba2rgb(vec4 rgba) {
    //return rgba.xyz * rgba.a + (1.0 - rgba.a) * 0.86;
    return rgba.xyz;
}
vec4 fxaa(sampler2D tex, vec2 fragCoord, vec2 resolution,
            vec2 v_rgbNW, vec2 v_rgbNE,
            vec2 v_rgbSW, vec2 v_rgbSE,
            vec2 v_rgbM) {
    vec4 color;
    mediump vec2 inverseVP = vec2(1.0 / resolution.x, 1.0 / resolution.y);
    vec4 texNW = COMPAT_TEXTURE(tex, v_rgbNW);
    vec4 texNE = COMPAT_TEXTURE(tex, v_rgbNE);
    vec4 texSW = COMPAT_TEXTURE(tex, v_rgbSW);
    vec4 texSE = COMPAT_TEXTURE(tex, v_rgbSE);
    vec4 texColor = COMPAT_TEXTURE(tex, v_rgbM);
    vec3 luma = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgba2rgb(texNW), luma);
    float lumaNE = dot(rgba2rgb(texNE), luma);
    float lumaSW = dot(rgba2rgb(texSW), luma);
    float lumaSE = dot(rgba2rgb(texSE), luma);
    float lumaM  = dot(rgba2rgb(texColor), luma);
    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    mediump vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) *
                          (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);

    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = min(vec2(FXAA_SPAN_MAX, FXAA_SPAN_MAX),
              max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),
              dir * rcpDirMin)) * inverseVP;

    vec4 texA = 0.5 * (
        COMPAT_TEXTURE(tex, fragCoord * inverseVP + dir * (1.0 / 3.0 - 0.5)) +
        COMPAT_TEXTURE(tex, fragCoord * inverseVP + dir * (2.0 / 3.0 - 0.5)));
    vec4 texB = texA * 0.5 + 0.25 * (
        COMPAT_TEXTURE(tex, fragCoord * inverseVP + dir * -0.5) +
        COMPAT_TEXTURE(tex, fragCoord * inverseVP + dir * 0.5));

    float lumaB = dot(rgba2rgb(texA), luma);
    if ((lumaB < lumaMin) || (lumaB > lumaMax))
        color = texA;
    else
        color = texB;
    return color;
}

void texcoords(vec2 fragCoord, vec2 resolution,
            out vec2 v_rgbNW, out vec2 v_rgbNE,
            out vec2 v_rgbSW, out vec2 v_rgbSE,
            out vec2 v_rgbM) {
    vec2 inverseVP = 1.0 / resolution.xy;
    v_rgbNW = (fragCoord + vec2(-1.0, -1.0)) * inverseVP;
    v_rgbNE = (fragCoord + vec2(1.0, -1.0)) * inverseVP;
    v_rgbSW = (fragCoord + vec2(-1.0, 1.0)) * inverseVP;
    v_rgbSE = (fragCoord + vec2(1.0, 1.0)) * inverseVP;
    v_rgbM = vec2(fragCoord * inverseVP);
}

vec4 apply(sampler2D tex, vec2 fragCoord, vec2 resolution) {
    mediump vec2 v_rgbNW;
    mediump vec2 v_rgbNE;
    mediump vec2 v_rgbSW;
    mediump vec2 v_rgbSE;
    mediump vec2 v_rgbM;

    //compute the texture coords
    texcoords(fragCoord, resolution, v_rgbNW, v_rgbNE, v_rgbSW, v_rgbSE, v_rgbM);

    //compute FXAA
    return fxaa(tex, fragCoord, resolution, v_rgbNW, v_rgbNE, v_rgbSW, v_rgbSE, v_rgbM);
}


void main() {
  vec2 fragCoord = gl_FragCoord.xy;
  vec4 color = apply(tex_input, fragCoord, resolution);

#if __VERSION__ >= 130
  FragColor = color;
#else
  gl_FragColor = color;
#endif
}
#endif

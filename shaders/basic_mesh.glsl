// VERTEX SHADER
#if defined(VERTEX)

uniform mat4x4 K;
uniform mat4x4 MV;
uniform mat4x4 M;
uniform float point_size;

in vec3 aPos;
in vec3 aColor;
in vec3 aNormal;

out lowp vec3 VertColor;
out highp vec4 FragPos;
out highp vec3 Normal;

void main()
{
    gl_PointSize = point_size;
    FragPos = MV * vec4(aPos.x, aPos.y, aPos.z, 1.0);
    gl_Position = K * FragPos;
    VertColor = aColor;
    Normal = normalize(mat3x3(M) * aNormal);
}

// FRAGMENT SHADER
#elif defined(FRAGMENT)

precision highp float;
in lowp vec3 VertColor;
in vec4 FragPos;
in vec3 Normal;

uniform bool unlit;
uniform vec3 camPos;

layout(location = 0) out lowp vec4 FragColor;
layout(location = 1) out float Depth;

void main()
{
    if (unlit) {
        FragColor = vec4(VertColor, 1);
    } else {
        // FIXME make these uniforms, whatever for now
        float ambient = 0.3;
        float specularStrength = 0.6;
        float diffuseStrength = 0.7;
        float diffuse2Strength = 0.2;
        vec3 lightDir = normalize(vec3(0.5, 0.2, 1));
        vec3 lightDir2 = normalize(vec3(-0.5, -1.0, -0.5));

        float diffuse = diffuseStrength * max(dot(lightDir, Normal), 0.0);
        float diffuse2 = diffuse2Strength * max(dot(lightDir2, Normal), 0.0);

        vec3 viewDir = normalize(camPos - vec3(FragPos));
        vec3 reflectDir = reflect(-lightDir, Normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        float specular = specularStrength * spec;

        FragColor = (ambient + diffuse + diffuse2 + specular) * vec4(VertColor, 1);
    }

    Depth = length(FragPos.xyz);
}

#endif

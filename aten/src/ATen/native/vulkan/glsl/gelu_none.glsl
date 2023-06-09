#version 450 core
#define PRECISION $precision
#define FORMAT    $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2)         uniform PRECISION restrict           Block {
  ivec4 size;
  float kAlpha; /* M_SQRT1_2 */
  float kBeta; /* M_2_PI */
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const vec4 inval = texelFetch(uInput, pos, 0);
    const vec4 toerf = inval * vec4(uBlock.kAlpha);
    const vec4 toerf4 = toerf * toerf * toerf * toerf;
    const vec4 toarctan = vec4(2.0) * toerf * (vec4(1.0) + toerf4);
    const vec4 arctanv = atan(toarctan);
    const vec4 erfv = vec4(uBlock.kBeta) * arctanv;
    const vec4 outval = inval * vec4(0.5) * (vec4(1.0) + erfv);
    imageStore(uOutput, pos, outval);
  }
}

attribute  vec3 aPosition;
attribute  vec3 aTexCoord;



vec4 Distort(vec4 p)
{
    vec2 v = p.xy / p.w;
    // Convert to polar coords:
    float radius = length(v);
    if (radius > 0)
    {
      float theta = atan(v.y,v.x);

      // Distort:
      radius = pow(radius, 0.8);

      // Convert back to Cartesian:
      v.x = radius * cos(theta);
      v.y = radius * sin(theta);
      p.xy = v.xy * p.w;
    }
    return p;
}

void main(void)
{
  vec4 V = vec4(aPosition, 1.0);
	vec4 P = gl_ModelViewProjectionMatrix * V;
  gl_Position = P;
}

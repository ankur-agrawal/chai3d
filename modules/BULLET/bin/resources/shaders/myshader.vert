attribute vec3 aPosition;
attribute vec4 aColor;

varying vec4 vColor;

void main(void)
{
  vec4 V = vec4(aPosition, 1.0);
	vec4 P = gl_ModelViewProjectionMatrix * V;
  gl_Position = P;
  vColor = aColor;
}

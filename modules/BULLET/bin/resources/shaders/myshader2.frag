//==============================================================================
/*
    Software License Agreement (BSD License)
    Copyright (c) 2003-2016, CHAI3D.
    (www.chai3d.org)
*/
//==============================================================================

//------------------------------------------------------------------------------

varying vec3 vLightVec;
varying vec3 vEyeVec;
varying vec2 vTexCoord;
varying vec4 vColor;

//------------------------------------------------------------------------------

uniform sampler2D uColorMap;
uniform sampler2D uNormalMap;
uniform float uInvRadius;

//------------------------------------------------------------------------------

/* Parameters from mt3d forums */
const vec2 LeftLensCenter = vec2(0.2863248, 0.5);
const vec2 RightLensCenter = vec2(0.7136753, 0.5);
const vec2 LeftScreenCenter = vec2(0.25, 0.5);
const vec2 RightScreenCenter = vec2(0.75, 0.5);
const vec2 Scale = vec2(0.1469278, 0.2350845);
const vec2 ScaleIn = vec2(4, 2.5);
const vec4 HmdWarpParam   = vec4(1, 0.22, 0.24, 0);


/* Main warp */
vec2 HmdWarp(vec2 in01, vec2 LensCenter)
{
	vec2 theta = (in01 - LensCenter) * ScaleIn; // Scales to [-1, 1]
	float rSq = theta.x * theta.x + theta.y * theta.y;
	vec2 rvector = theta * (HmdWarpParam.x + HmdWarpParam.y * rSq +
		HmdWarpParam.z * rSq * rSq +
		HmdWarpParam.w * rSq * rSq * rSq);
	return LensCenter + Scale * rvector;
}


void main (void)
{
    float distSqr = dot(vLightVec, vLightVec);
    float att = clamp(1.0 - uInvRadius * sqrt(distSqr), 0.0, 1.0);
    vec3 lVec = vLightVec * inversesqrt(distSqr);

    vec3 vVec = normalize(vEyeVec);

    vec4 base = texture2D(uColorMap, vTexCoord);

    vec3 bump = normalize( texture2D(uNormalMap, vTexCoord).xyz * 2.0 - 1.0);

    vec4 vAmbient = gl_FrontMaterial.ambient;

    float diffuse = max( dot(lVec, bump), 0.0 );

    vec4 vDiffuse = gl_FrontMaterial.diffuse *
                    diffuse;

    float specular = pow(clamp(dot(reflect(-lVec, bump), vVec), 0.0, 1.0),
                     gl_FrontMaterial.shininess );

    vec4 vSpecular = gl_FrontMaterial.specular *
                     specular;

    gl_FragColor = base;
}

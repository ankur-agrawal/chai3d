//==============================================================================
/*
    Software License Agreement (BSD License)
    Copyright (c) 2003-2016, CHAI3D
    (www.chai3d.org)

    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following
    disclaimer in the documentation and/or other materials provided
    with the distribution.

    * Neither the name of CHAI3D nor the names of its contributors may
    be used to endorse or promote products derived from this software
    without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

    \author    <http://www.aimlab.wpi.edu>
    \author    Ankur Agrawal
    \version   3.2.0 $Rev: 2015 $
*/
//==============================================================================

//------------------------------------------------------------------------------
#ifndef CBulletEndoscopeH
#define CBulletEndoscopeH
//------------------------------------------------------------------------------
#include "chai3d.h"
#include "CBullet.h"
#include "CBulletMultiMesh.h"
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
namespace chai3d {
//------------------------------------------------------------------------------

//==============================================================================
/*!
    \file       CBulletEndoscope.h

    \brief
    <b> Bullet Module </b> \n
    Bullet Endoscope Object.
*/
//==============================================================================

//==============================================================================
/*!
    \class      cBulletEndoscope
    \ingroup    Bullet

    \brief
    This class implements a Bullet dynamic endoscope.

    \details
    cBulletEndoscope models a dynamic endoscope.
*/
//==============================================================================

class cBulletEndoscope : public cBulletGenericObject{
  //--------------------------------------------------------------------------
  // CONSTRUCTOR & DESTRUCTOR:
  //--------------------------------------------------------------------------
public:
  cBulletEndoscope(cBulletWorld* a_world,
                   cVector3d a_rcm_pos, cMatrix3d a_rcm_rot,
                   double yaw_joint, double pitch_joint, double insertion_length, double roll_joint,
                   std::string a_objName = "endoscope");

  ~cBulletEndoscope(){
    if (m_rosObjPtr)
    {
      delete m_rosObjPtr.get();
    }
  };

  void setCameraRotFromJoints();
  void setCameraPosFromJoints();
  void setJointsFromCameraRot();
  virtual void updateCmdFromROS(double dt);
  void updatePositionFromDynamics();
  cMatrix3d getCommandedRot();

  cCamera* m_camera;
  cMatrix3d cur_rot_mat, cmd_rot_mat, cmd_rot_mat_last;
  cMatrix3d oculus_ecm_rot_mat;
  std::vector<double> joint_angles;
  cVector3d rcm_pos;
  cMatrix3d rcm_rot;
  cVector3d camera_pos;
  cMatrix3d camera_rot;

};
}

#endif

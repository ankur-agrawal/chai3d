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

#include "CBulletEndoscope.h"
#include <string.h>


namespace chai3d{

cBulletEndoscope::cBulletEndoscope(cBulletWorld* a_world,
                 cVector3d a_rcm_pos, cMatrix3d a_rcm_rot,
                 double yaw_joint, double pitch_joint, double insertion_length, double roll_joint,
                 std::string a_objName) : cBulletGenericObject(a_world, a_objName)
{
  m_camera = new cCamera(a_world);
  a_world->addChild(m_camera);
  rcm_pos = a_rcm_pos;
  rcm_rot = a_rcm_rot;

  // set stereo mode
  m_camera->setStereoMode(C_STEREO_PASSIVE_LEFT_RIGHT);


  // set the near and far clipping planes of the camera
  m_camera->setClippingPlanes(0.1, 100.0);

  // set stereo eye separation and focal length (applies only if stereo is enabled)
  m_camera->setStereoEyeSeparation(0.02);
  m_camera->setStereoFocalLength(20.0);

  m_camera->setFieldViewAngleDeg(30);
  // set vertical mirrored display mode
  m_camera->setMirrorVertical(false);

  joint_angles.resize(4);
  joint_angles[0] = yaw_joint;
  joint_angles[1] = pitch_joint;
  joint_angles[2] = insertion_length;
  joint_angles[3] = roll_joint;

  setCameraRotFromJoints();
  setCameraPosFromJoints();

  oculus_ecm_rot_mat = cMatrix3d(0,1,0,0,0,1,1,0,0);
  cmd_rot_mat=cMatrix3d(1,0,0,0,1,0,0,0,1);
  cmd_rot_mat_last=cMatrix3d(1,0,0,0,1,0,0,0,1);
  cur_rot_mat = m_camera->getLocalRot();
  cmd_rot_mat.mul(oculus_ecm_rot_mat);
  cmd_rot_mat_last.mul(oculus_ecm_rot_mat);
  // setMass(0.5);
  // estimateInertia();
  // buildDynamicModel();
}

void cBulletEndoscope::setCameraRotFromJoints()
{
  cMatrix3d camera_in_rcm_rot;
  camera_in_rcm_rot(0,0) = -cos(joint_angles[1])*sin(joint_angles[0]);
  camera_in_rcm_rot(0,1) = cos(joint_angles[0])*cos(joint_angles[3]) - sin(joint_angles[0])*sin(joint_angles[1])*sin(joint_angles[3]);
  camera_in_rcm_rot(0,2) = cos(joint_angles[0])*sin(joint_angles[3]) + cos(joint_angles[3])*sin(joint_angles[0])*sin(joint_angles[1]);
  camera_in_rcm_rot(1,0) = sin(joint_angles[1]);
  camera_in_rcm_rot(1,1) = -cos(joint_angles[1])*sin(joint_angles[3]);
  camera_in_rcm_rot(1,2) = cos(joint_angles[1])*cos(joint_angles[3]);
  camera_in_rcm_rot(2,0) = cos(joint_angles[1])*cos(joint_angles[0]);
  camera_in_rcm_rot(2,1) = sin(joint_angles[0])*cos(joint_angles[3]) + cos(joint_angles[0])*sin(joint_angles[1])*sin(joint_angles[3]);
  camera_in_rcm_rot(2,2) = sin(joint_angles[0])*sin(joint_angles[3]) - cos(joint_angles[3])*cos(joint_angles[0])*sin(joint_angles[1]);
  camera_rot = cMul(rcm_rot, camera_in_rcm_rot);
  // std::cout << camera_rot.str() << '\n';
  m_camera->setLocalRot(camera_rot);
}

void cBulletEndoscope::setCameraPosFromJoints()
{
  cVector3d camera_pos_in_rcm;

  camera_pos_in_rcm(0) = cos(joint_angles[1])*sin(joint_angles[0])*joint_angles[2];
  camera_pos_in_rcm(1) = -sin(joint_angles[1])*joint_angles[2];
  camera_pos_in_rcm(2) = -cos(joint_angles[1])*cos(joint_angles[0])*joint_angles[2];

  camera_pos = rcm_pos + cMul(rcm_rot, camera_pos_in_rcm);
  // std::cout << camera_pos << '\n';
  m_camera->setLocalPos(camera_pos);
}

void cBulletEndoscope::updateInsertion(double dlen)
{
  joint_angles[2] = joint_angles[2] + dlen;
  if (joint_angles[2] > 24)
  {
    joint_angles[2] = 24;
  }
  if (joint_angles[2] < 0)
  {
    joint_angles[2] = 0;
  }
  setJointsFromCameraRot();
  setCameraPosFromJoints();
  setCameraRotFromJoints();
}

void cBulletEndoscope::updateYaw(double dlen)
{
  joint_angles[0] = joint_angles[0] + dlen;
  if (joint_angles[0] > 1.5)
  {
    joint_angles[0] = 1.5;
  }
  if (joint_angles[0] < -1.5)
  {
    joint_angles[0] = -1.5;
  }
  setCameraPosFromJoints();
  setCameraRotFromJoints();
}

void cBulletEndoscope::updatePitch(double dlen)
{
  joint_angles[1] = joint_angles[1] + dlen;
  if (joint_angles[1] > 1.5)
  {
    joint_angles[1] = 1.5;
  }
  if (joint_angles[1] < -1.5)
  {
    joint_angles[1] = -1.5;
  }
  setCameraPosFromJoints();
  setCameraRotFromJoints();
}

void cBulletEndoscope::setJointsFromCameraRot()
{
  cMatrix3d camera_in_rcm_rot;
  camera_in_rcm_rot = cMul(cTranspose(rcm_rot), camera_rot);
  joint_angles[1] = atan2(camera_in_rcm_rot(1,0), sqrt(pow(camera_in_rcm_rot(0,0),2)+pow(camera_in_rcm_rot(2,0),2)));
  joint_angles[0] = atan2(-camera_in_rcm_rot(0,0)/cos(joint_angles[1]), camera_in_rcm_rot(2,0)/cos(joint_angles[1]));
  joint_angles[3] = atan2(-camera_in_rcm_rot(1,1)/cos(joint_angles[1]), camera_in_rcm_rot(1,2)/cos(joint_angles[1]));
  joint_angles[2] = joint_angles[2];
  if (abs(joint_angles[1])>1.5708)
    std::cout << joint_angles[0]  << '\t' << joint_angles[1] << '\t' << joint_angles[2]  << '\t' << joint_angles[3]  << '\t'<< '\n';
}

void cBulletEndoscope::updateCmdFromROS(double dt){
  static int first=0;
  first++;
  cVector3d cur_pos, cmd_pos, rot_axis;
  cQuaternion cur_rot, cmd_rot;
  if (first==1)
  {
    cmd_rot.x = m_rosObjPtr->m_afCmd.qx;
    cmd_rot.y = m_rosObjPtr->m_afCmd.qy;
    cmd_rot.z = m_rosObjPtr->m_afCmd.qz;
    cmd_rot.w = m_rosObjPtr->m_afCmd.qw;
    cmd_rot.toRotMat(cmd_rot_mat_last);
    cmd_rot_mat_last.mul(oculus_ecm_rot_mat);
  }
    if (m_rosObjPtr.get() != nullptr){
        m_rosObjPtr->update_af_cmd();
        cVector3d force, torque;
        if (m_rosObjPtr->m_afCmd.pos_ctrl){


            cur_pos=m_camera->getLocalPos();
            cur_rot_mat=m_camera->getLocalRot();

            cmd_rot.x = m_rosObjPtr->m_afCmd.qx;
            cmd_rot.y = m_rosObjPtr->m_afCmd.qy;
            cmd_rot.z = m_rosObjPtr->m_afCmd.qz;
            cmd_rot.w = m_rosObjPtr->m_afCmd.qw;
            cmd_rot.normalize();
            cmd_rot.toRotMat(cmd_rot_mat);

            cmd_rot_mat.mul(oculus_ecm_rot_mat);

            // std::cout << cMul(cur_rot_mat, cMul(cTranspose(cmd_rot_mat_last), cmd_rot_mat)).str() << '\n';

            camera_rot = cMul(cur_rot_mat, cMul(cTranspose(cmd_rot_mat_last), cmd_rot_mat));

            setJointsFromCameraRot();
            setCameraRotFromJoints();
            setCameraPosFromJoints();

            cmd_rot_mat_last=cmd_rot_mat;
        }
        else{
            if (m_rosObjPtr->m_afCmd.J_cmd.size() == 4)
            {
                joint_angles[0] = m_rosObjPtr->m_afCmd.J_cmd[0];
                joint_angles[1] = m_rosObjPtr->m_afCmd.J_cmd[1];
                joint_angles[2] = m_rosObjPtr->m_afCmd.J_cmd[2];
                joint_angles[3] = m_rosObjPtr->m_afCmd.J_cmd[3];
                setCameraPosFromJoints();
                setCameraRotFromJoints();
            }
        }
        // addExternalForce(force);
        // addExternalTorque(torque);
    }
}

void cBulletEndoscope::updatePositionFromDynamics()
{
    if (m_camera)
    {
        cVector3d pos;
        cMatrix3d rot_mat;
        cQuaternion quat;

        pos=m_camera->getLocalPos();
        rot_mat=m_camera->getLocalRot();
        quat.fromRotMat(rot_mat);
        quat.normalize();

        // update Transform data for m_rosObj
        if(m_rosObjPtr.get() != nullptr)
        {
          m_rosObjPtr->cur_position(pos.x(),pos.y(),pos.z());
          m_rosObjPtr->cur_orientation(quat.x, quat.y, quat.z, quat.w);
        }
    }
}
cMatrix3d cBulletEndoscope::getCommandedRot()
{
  return cMul(cur_rot_mat, cMul(cTranspose(cmd_rot_mat_last), cmd_rot_mat));
}


}

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
                 cVector3d rcm_pos, cMatrix3d rcm_ori,
                 cVector3d camera_pos, cMatrix3d camera_ori,
                 std::string a_objName) : cBulletGenericObject(a_world, a_objName)
{
  m_camera = new cCamera(a_world);
  a_world->addChild(m_camera);
  m_camera->setLocalPos(camera_pos);
  m_camera->setLocalRot(camera_ori);

  // set stereo mode
  m_camera->setStereoMode(C_STEREO_PASSIVE_LEFT_RIGHT);


  // set the near and far clipping planes of the camera
  m_camera->setClippingPlanes(0.01, 10.0);

  // set stereo eye separation and focal length (applies only if stereo is enabled)
  m_camera->setStereoEyeSeparation(0.02);
  m_camera->setStereoFocalLength(2.0);

  m_camera->setFieldViewAngleDeg(30);
  // set vertical mirrored display mode
  m_camera->setMirrorVertical(false);

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

            // cMatrix3d cur_rot_mat, cmd_rot_mat;
            btTransform b_trans;
            double rot_angle;
            double K_lin = 10, B_lin = 1;
            double K_ang = 5;

            // m_bulletRigidBody->getMotionState()->getWorldTransform(b_trans);
            // cur_pos.set(b_trans.getOrigin().getX(),
            //             b_trans.getOrigin().getY(),
            //             b_trans.getOrigin().getZ());
            //
            // cur_rot.x = b_trans.getRotation().getX();
            // cur_rot.y = b_trans.getRotation().getY();
            // cur_rot.z = b_trans.getRotation().getZ();
            // cur_rot.w = b_trans.getRotation().getW();
            // cur_rot.toRotMat(cur_rot_mat);

            cur_pos=m_camera->getLocalPos();
            cur_rot_mat=m_camera->getLocalRot();

            // cmd_pos.set(m_rosObjPtr->m_afCmd.px,
            //             m_rosObjPtr->m_afCmd.py,
            //             m_rosObjPtr->m_afCmd.pz);

            cmd_rot.x = m_rosObjPtr->m_afCmd.qx;
            cmd_rot.y = m_rosObjPtr->m_afCmd.qy;
            cmd_rot.z = m_rosObjPtr->m_afCmd.qz;
            cmd_rot.w = m_rosObjPtr->m_afCmd.qw;
            cmd_rot.normalize();
            cmd_rot.toRotMat(cmd_rot_mat);

            cmd_rot_mat.mul(oculus_ecm_rot_mat);
            // m_dpos_prev = m_dpos;
            // m_dpos = cmd_pos - cur_pos;
            // m_ddpos = (m_dpos - m_dpos_prev)/dt;
            // m_drot = cMul(cTranspose(cur_rot_mat), cmd_rot_mat);
            // m_drot.toAxisAngle(rot_axis, rot_angle);
            //
            // force = K_lin * m_dpos + B_lin * m_ddpos;
            // torque = cMul(K_ang * rot_angle, rot_axis);
            // cur_rot_mat.mul(torque);

            // std::cout << cMul(cur_rot_mat, cMul(cTranspose(cmd_rot_mat_last), cmd_rot_mat)).str() << '\n';

            // m_camera->setLocalRot(cMul(cur_rot_mat, cMul(oculus_ecm_rot_mat,cMul(cTranspose(cmd_rot_mat_last), cmd_rot_mat))));
            m_camera->setLocalRot(cMul(cur_rot_mat, cMul(cTranspose(cmd_rot_mat_last), cmd_rot_mat)));
            // m_camera->setLocalRot(cMul(oculus_ecm_rot_mat,cMul(cur_rot_mat, cMul(cTranspose(cmd_rot_mat_last), cmd_rot_mat))));

            cmd_rot_mat_last=cmd_rot_mat;
            // m_camera->setLocalRot(cMul(oculus_ecm_rot_mat,cmd_rot_mat));
        }
        else{

            force.set(m_rosObjPtr->m_afCmd.Fx,
                      m_rosObjPtr->m_afCmd.Fy,
                      m_rosObjPtr->m_afCmd.Fz);
            torque.set(m_rosObjPtr->m_afCmd.Nx,
                       m_rosObjPtr->m_afCmd.Ny,
                       m_rosObjPtr->m_afCmd.Nz);
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

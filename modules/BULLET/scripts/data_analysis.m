clear
clc
data = textread('/home/ankur/chai3d/modules/BULLET/bin/resources/data/0202.csv');

figure;
iter =1;
a = 0.001;
% for i=1:size(data,1)
%     if i==1
%         gripper_pos_filtered(i,:) = data(i,37:39);
%     else
%         gripper_pos_filtered(i,:) = a*data(i,37:39)+(1-a)*data(i-1,37:39);
%     end
%     gripper_pos(i,:) = norm(gripper_pos_filtered(i,:));
% end
% 
% for i = 1:1000:size(data,1)
% if  iter ==1
% gripper_velocity(iter,:) = [0,0,0];
% gripper_acc(iter,:) =[0,0,0];
% else
% gripper_velocity(iter,:) = (gripper_pos_filtered(i,:)-gripper_pos_filtered(i-1000,:))/(data(i,1)-data(i-1000,1));
% gripper_acc(iter,:) =(gripper_velocity(iter,:)-gripper_velocity(iter-1,:))/(data(i,1)-data(i-1000,1));
% end
% 
% gripper_speed(iter,1) = norm(gripper_velocity(iter,:));
% gripper_accel(iter,1) = norm(gripper_acc(iter,:));
% 
% iter = iter+1;
% end
% plot(data(1:1000:end,1), data(1:1000:end,36), data(1:1000:end,1), gripper_speed, data(1:1000:end,1), gripper_accel)
plot(data(:,1),data(:,36))
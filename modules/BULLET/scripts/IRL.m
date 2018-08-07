clear
clc
data = textread('/home/ankur/chai3d/modules/BULLET/bin/resources/data/0103.csv');

a = 0.005;
for i=1:size(data,1)
    if i==1
        gripper_pos_filtered(i,:) = data(i,37:39);
        joint_angles_filtered = data(:,2:5);
    else
        gripper_pos_filtered(i,:) = a*data(i,37:39)+(1-a)*data(i-1,37:39);
        joint_angles_filtered(i,:) = a*data(i,2:5)+(1-a)*joint_angles_filtered(i-1,:);
    end
    gripper_pos(i,:) = norm(gripper_pos_filtered(i,:));
end

joint_angles = data(:,2:5);

% plot(data(:,1),joint_angles(:,1),data(:,1),joint_angles(:,2),data(:,1),joint_angles(:,4))

subplot(3,1,1)
plot(data(:,1),joint_angles(:,1),data(:,1),joint_angles_filtered(:,1))
subplot(3,1,2)
plot(data(:,1),joint_angles(:,2),data(:,1),joint_angles_filtered(:,2))
subplot(3,1,3)
plot(data(:,1),joint_angles(:,4),data(:,1),joint_angles_filtered(:,4))
% plot(data(:,1),joint_angles_filtered(:,1),data(:,1),joint_angles_filtered(:,2),data(:,1),joint_angles_filtered(:,4))
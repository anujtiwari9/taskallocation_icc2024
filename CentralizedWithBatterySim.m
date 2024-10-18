clc
clear all

%Initial robot Locations
bot_locs = zeros(25,2);

b=1;
b_dash = 0;
while(b<=20 & b_dash<=9)
        bot_locs(b,:) = [7+8*b_dash, 2];
        bot_locs(b+1,:) = [7+8*b_dash, 4];
        b = b+2;
        b_dash = b_dash +1;
end
b_dash = 0;
while(b<=25 & b_dash<=4)
        bot_locs(b,:) = [27+5*b_dash, 6];
        b = b+1;
        b_dash = b_dash +1;
end

%Number of tasks done by each robot: normal and charging
Tasks = zeros(25,2);

%Initializing Final Task Locations of all robots
final_task_locs = zeros(25,2);

%Initializing Initial Battery Level of all robots
battery = 100*ones(size(bot_locs,1),1);

%Number of tasks per order set, assuming a total of 5 set of orders came
num_tasks_list = [50, 60, 70, 60, 50];
%Orders Come in an intervals of 5000 pose steps
time_steps = [1,5000,10000,15000,20000,25000];
%Initializing a poses cell
locations_dash = cell(size(bot_locs,1),1);
for i = 1:1:size(bot_locs,1)
    locations_dash{i} = zeros(0,3);
end


for OrderSet = 1:1:5
    fprintf("OrderSet %d has arrived, Centralized with battery consideration Task Allotment has Began.\n", OrderSet);
    num_tasks = num_tasks_list(OrderSet);
    [locations,final_battery,map, Tasks] = DecentralizedTwoLayerTaskAllocaton(bot_locs,num_tasks,final_task_locs,battery, Tasks);
    for i = 1:1:size(locations,1)
        final_task_locs(i, :) = [locations{i}(size(locations{i},1),1), locations{i}(size(locations{i},1),2)];
    end
    %final_task_locs = round(final_task_locs);
    final_task_locs = double(ceil(final_task_locs));
    final_task_locs = checkOcc(final_task_locs);
    for i = 1:1:size(locations,1)
        if ~isequal(locations_dash{i},locations{i})
            locations_dash{i} = [locations_dash{i}; locations{i}];
        else
            locations_dash{i} = locations{i};
        end
    end
    battery = final_battery;
    m = FindMaxTimeStep(locations_dash);
    if m >= time_steps(6)
        time_steps(6) = m;
    end
    Tstart = time_steps(OrderSet);
    Tend = time_steps(OrderSet+1);
    fprintf("Simulation Has Started.\n");
    [bot_locs] = Visualization(map,locations_dash,Tstart,Tend);
end


fprintf("Simulation of Task Allocation of 5 set of Orders has been Completed.\n");


%Distance calculation of all robots
Robot_distances = dCal(locations_dash, bot_locs);
%Charging Visits by each robot
Charge_visits = Tasks(:,2);

%Plotting results
%Distance travelled by each robot
robot_numbers = 1:25;
figure;
bar(robot_numbers, Robot_distances);
xlabel('Robot Number');
ylabel('Distance Traveled');
title('Distance Traveled by Each Robot(Centralized with Battery)');
grid on;
xlim([0.5, 25.5]);
ylim([0 15000]);
xticks(1:25);

%number of times charge station visited by each robot
figure;
bar(robot_numbers, Tasks(:,2));
xlabel('Robot Number');
ylabel('Number of times went for charging');
title('Number of times charge station visited by Each Robot(Centralized with Battery)');
grid on;
xlim([0.5, 25.5]);
ylim([0 10]);
xticks(1:25);


%%%%%Functions Used%%%%%%%
function [locations,final_battery,map, Tasks] = DecentralizedTwoLayerTaskAllocaton(bot_locs,num_tasks,final_task_locs,battery,Tasks)
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %clustering based on proximity
    min_diff_between_centroids = 10;
    C = 0;
    
    for k=2:1:10
    
        if (C ~= 0)
    
            %calculating min distance between any two points
            D = pdist(C);
            D = squareform(D); 
            D = D + max(D(:))*eye(size(D)); % ignore zero distances on diagonals 
            [minD,temp] = min(D(:));
        
            min_dist_btw_centroids = minD;
        
            if (min_dist_btw_centroids <= min_diff_between_centroids)
        
                break;
        
            end
    
        end
    
    
        [idx, C] = kmeans(bot_locs, k);
        num_of_clusters = 1;
    
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %STEP 2
    %code to divide pod_locs into different zones based on zone centroids
    %get the task-zone priority matrix which is NXM, M being the number of
    %zones
    
    %Initializing pod locations according to warehouse plan.
    pod_locs = zeros(768,3);%PX3 matrix
    x_stacks = 16;%number of stacks of pods in the x direction
    y_stacks = 4;%number of stacks of pods in the y direction
    z = 1;%used for filling pod locations
    while(z<=1)
        for k = 0:y_stacks-1
           for i = 0:x_stacks-1
              for j = 0:5
                  pod_locs(z,:) = [z,1+10+5*i, 15 + 2*j + 13*k];
                  pod_locs(z+1,:) = [z+1,1+10+5*i+2, 15 + 2*j + 13*k];
                  z = z+2;
              end
           end
        end
    end
    
    %Assume num_tasks pods are selected after order batch to robot task conversion.
    %Those pods are selected randomly for now.
    num_pods = size(pod_locs, 1);
    random_indices = randperm(num_pods, num_tasks);
    task_pods = pod_locs(random_indices, :);%These are the selected pods
    
    %This counts list gives the number of robots in each zone.
    unique_values = unique(idx);
    counts = zeros(size(unique_values));
    for i = 1:numel(unique_values)
        counts(i) = sum(idx == unique_values(i));
    end
    
    %task-zone priority matrix
    tz = zeros(size(task_pods,1),num_of_clusters);
    for i = 1:size(task_pods,1)
        for j = 1:num_of_clusters
            tz(i,j) = p(counts(j), size(bot_locs,1), C(j,1), C(j,2), task_pods(i,2), task_pods(i,3));
        end
    end
    
    %zone_numbers is a list that gives the zone alloted to each task.
    [max_priorities, zone_numbers] = find_max_priority(tz);
    zone_numbers = zone_numbers';
    
    %robotzone matrix
    %The rbzones matrix basically gives the robot number, the zone it belongs to
    %and its battery level.
    robots = reshape(1:size(bot_locs,1),[size(bot_locs,1),1]);
    rbzones = [robots,idx,battery];
    
    %zone_lists is a cell whose each element gives a list which has the robot
    %IDs that are part of each zone, each element in the cell corresponds to a
    %zone.
    unique_zones = unique(rbzones(:,2));
    zone_lists = cell(size(unique_zones));
    for i = 1:numel(unique_zones)
        zone = unique_zones(i);
        
        % Find the indices of robots in the current zone
        zone_indices = find(rbzones(:,2) == zone);
        
        % Extract the robot IDs for the current zone
        zone_robot_ids = rbzones(zone_indices,1);
        
        % Store the robot IDs in the corresponding cell of the zone_lists
        zone_lists{i} = zone_robot_ids;
    end
    
    %This prints out the task/pod number assigned to each zone
    for i = 1:size(task_pods,1)
        fprintf('task%d is allotted to zone%d.\n', task_pods(i,1), zone_numbers(i,1));
    end
    %fprintf("Zone wise allotment of Current Set of OrderTasks are complete.\n");
    %Initializing Pick station locations according to warehouse layout.
    pick1 = [7,77];
    pick2 = [23,77];
    pick3 = [38,77];
    picks = [pick1;pick2;pick3];%It is a list that contains the x-y coordinates of each pick station in each row.
    
    %Initializing charge station location according to warehouse layout.
    %Initial robot Locations = charge station locations
    charge_locs = zeros(25,2);
    
    b=1;
    b_dash = 0;
    while(b<=20 & b_dash<=9)
            charge_locs(b,:) = [7+8*b_dash, 2];
            charge_locs(b+1,:) = [7+8*b_dash, 4];
            b = b+2;
            b_dash = b_dash +1;
    end
    b_dash = 0;
    while(b<=25 & b_dash<=4)
            charge_locs(b,:) = [27+5*b_dash, 6];
            b = b+1;
            b_dash = b_dash +1;
    end
    %chargeStations = [1,97,7;2,97,23;3,97,38];%1st column is station number and 2nd and 3rd columns are its x and y coordinates.
    Robot_charge_stations = charge_locs;
    %for i = 1:1:size(rbzones,1)
        %a = randi(size(chargeStations,1));
        %Robot_charge_stations(i,:) = chargeStations(a,2:3);
    %end
    rbzones = [rbzones,Robot_charge_stations];
    
    
    %Assigning randomn pick stations to each task, this is encoded in the pick
    %stations list.
    pick_stations = zeros(size(task_pods,1),2);
    for i = 1:size(task_pods,1)
        rand_index = randi(size(picks,1)); 
        pick_stations(i,:) = picks(rand_index, :);
    end
    
    %Concatenating the task_pods, pick_stations and zone_numbers lists to form
    %a new task_pods matrix that has task ID, Its x and y location, its pick
    %station x and y location and its allocated zone.
    task_pods = [task_pods, pick_stations,zone_numbers];
    
    %A task_pods_dash matrix is created for iterating purposes while task
    %allotment in each zone.
    task_pods_dash = task_pods;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Task allocation within each zone
    bid = cell(num_of_clusters,1);% A bid cell, whose each element corresponds to a zone and is a matrix whose rows represent the robots in that zone and columns represents all the tasks(Not just the tasks of that zone).
    %Threshold battery level, this is made assuming that on an average any
    %robot in any part ot the warehouse would require 20 percent charge to go
    %to the charging station.
    Threshold = 20;
    %In the first set of orders bot locs dash are their current locations
    %but once the first set of orders are done and second set arrives then
    %bot locs dash should be the location of the final task of that robot
    %in first set of orders.
    if final_task_locs == zeros(25,2)
        bot_locs_dash = bot_locs;
    else
        bot_locs_dash = final_task_locs;
    end
    %Creating the number of rows and columns of each element in the bid cell.
    rows = counts';
    columns = [];
    for i = 1:1:size(zone_lists,1)
        z = size(task_pods,1);
        columns(end+1) = z;
    end
    %Creating each element/matrix in the bid cell and Setting all the bid elements to a very high value.
    for i = 1:num_of_clusters
        bid{i} = 10000*ones(rows(i), columns(i));
    end
    
    %Creating a cell which keeps track of all the tasks accepted by each robot.
    rbTask = cell(size(bot_locs,1),1);
    
    
    for i = 1:1:size(zone_lists,1)%Running through each zone for second layer of task allotment.
        while true
            zoneIndices = find(task_pods_dash(:, 6) == i);%This gives a list with tasks whose zone is the zone we are currently allocating tasks at.
            %If zoneIndices is empty, then we break out of the while loop
            %because second level of allotment is not needed there.
            if isempty(zoneIndices)
                break;
            end
            temp_charge_level = 100*ones(size(zone_lists{i},1),size(task_pods,1));%This is a matrix for the zone we are currently looking at, whose rows have the robots in the zone and columns are the total tasks and each element is set to 100.
            for j = 1:1:size(zone_lists{i},1)%Running through each robot in the zone.
                for k = 1:1:size(task_pods,1)%Running through all the tasks
                      TId = task_pods(k,1);%find the pod ID of the task under consideration
                      T = find(task_pods_dash(:, 1) == TId);%find the row number in which that ID is present in task_pods_dash, This is done beacuse task_pods_dash Reduces in size after every iteration, so kth element of task_pods will not be same as kth element of task_pods_dash, but since the ID will be same, we check for that ID in task_pods_dash.
                      if i == task_pods_dash(T,6)%See if that task is part of the zone we are currently at.
                          %Current robot location.
                          bot_loc_x = bot_locs_dash(zone_lists{i}(j,1),1);
                          bot_loc_y = bot_locs_dash(zone_lists{i}(j,1),2);
                          %Location of task/pod under consideration
                          task_pod_x = task_pods(k,2);
                          task_pod_y = task_pods(k,3);
                          %Location of the pick station of the task/pod under
                          %consideration
                          pick_pod_x = task_pods(k,4);
                          pick_pod_y = task_pods(k,5);
                          %Current battery level of robot
                          rb_battery = rbzones(zone_lists{i}(j,1),3);
                          [bid{i}(j,k),temp_charge_level(j,k)] = cost(bot_loc_x,bot_loc_y,task_pod_x,task_pod_y,pick_pod_x,pick_pod_y,rb_battery);%Finds the bid for the robot-task pair under consideration and finds the charge left for that robot if it performs that task, assigns the found bid to the bid cell and the found charge level to the temp_charge_level matrix.
                      end
                end
            end
            [min_bid, rowIdx, colIdx] = findMinAndIndices(bid{i});%finding the minimum bid and the corresponding robot task pair in the zone under consideration after the current run through the zone.
            if temp_charge_level(rowIdx,colIdx) <= Threshold
                rbTask{zone_lists{i}(rowIdx,1)}(end+1) = 0;%task number 0 implies it is a charging task.
                rbzones(zone_lists{i}(rowIdx,1),3) = 100;%Change the charge of the robot to 100 because it went for charging.
                bot_locs_dash(zone_lists{i}(rowIdx,1),1) = rbzones(zone_lists{i}(rowIdx,1),4);%Change the x location of the robot to charge station location.
                bot_locs_dash(zone_lists{i}(rowIdx,1),2) = rbzones(zone_lists{i}(rowIdx,1),5);%Change the y location of the robot to charge station location.
                Tasks(zone_lists{i}(rowIdx,1), 2) = Tasks(zone_lists{i}(rowIdx,1), 2) + 1;
                fprintf("Robot%d has been alloted a charging task and cannot be alloted task%d.\n",rbzones(zone_lists{i}(rowIdx,1),1),task_pods(colIdx,1));
                continue
            end
            rbzones(zone_lists{i}(rowIdx,1),3) = temp_charge_level(rowIdx,colIdx);%Change the charge of the robot that got allotted the task.
            bot_locs_dash(zone_lists{i}(rowIdx,1),1) = task_pods(colIdx,2);%Change the x location of the robot that got allotted the task to the Task location.
            bot_locs_dash(zone_lists{i}(rowIdx,1),2) = task_pods(colIdx,3);%Change the y location of the robot that got allotted the task to the Task location.
            Tasks(zone_lists{i}(rowIdx,1), 1) = Tasks(zone_lists{i}(rowIdx,1), 1) + 1;
            fprintf('Task%d is allotted to Robot%d in zone%d.\n',task_pods(colIdx,1),rbzones(zone_lists{i}(rowIdx,1),1),i);
            %Adding the allotted task ID to the allotted robot.
            rbTask{zone_lists{i}(rowIdx,1)}(end+1) = task_pods(colIdx,1);
            %Remove the allotted task from task_pods_dash list.
            TaskId = task_pods(colIdx,1);
             t = find(task_pods_dash(:, 1) == TaskId);
             task_pods_dash(t,:) = [];
             %Set the bid matrix of the zone back to high values to avoid
             %the bids of this run getting mixed up with the bids of the
             %next run through the same zone.
             bid{i} = 10000*ones(rows(i), columns(i));
        end
    end 
    final_battery = rbzones(:,3);
    %fprintf("Second Layer of Task allotment is complete.\n");
    fprintf("Collecting poses of Planned Path of Robots for alloted tasks.....\n");
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%Path Planning - Pose collection of A* paths of each robot
    if final_task_locs == zeros(25,2)
        bot_locs_planning = bot_locs;
    else
        bot_locs_planning = final_task_locs;
    end
    task_pods_planning = [task_pods(:,1:3),task_pods(:,4:5),task_pods(:,2:3)];
    locations = cell(size(bot_locs,1),1);
    for i = 1:1:size(bot_locs,1)
        locations{i} = zeros(0,3);
    end
    
    for i = 1:1:size(bot_locs,1)
        if isempty(rbTask{i})
            locations{i} = [bot_locs_planning(i,:),0];
        else
            for j = 1:1:size(rbTask{i},2)
                tID = rbTask{i}(j);
                if tID ~= 0
                   b = 2;
                   t = find(task_pods(:, 1) == tID);
                   while(b<=6)
                      start = [bot_locs_planning(i,1),bot_locs_planning(i,2),0];
                      stop = [task_pods_planning(t,b),task_pods_planning(t,b+1),0];
                      if ~isequal(start,stop)
                          [map,poses] = Path_Planner(start,stop);
                          locations{i} = [locations{i};poses];
                      end
                      bot_locs_planning(i,:) = [task_pods_planning(t,b),task_pods_planning(t,b+1)];
                      b = b+2;
                   end
                else
                    start = [bot_locs_planning(i,:),0];
                    stop = [rbzones(i,4:5),0];
                    if ~isequal(start,stop)
                        [map,poses] = Path_Planner(start,stop);
                        locations{i} = [locations{i};poses];
                        bot_locs_planning(i,:) = rbzones(i,4:5);
                    end
                end
            end
        end
    end
    
end

function [bot_locs] = Visualization(map,locations,Tstart,Tend)
     sizes = zeros(1,size(locations,1));
     for i = 1:1:size(locations,1)
          sizes(i) = size(locations{i},1);
     end
    for i = Tstart:100:Tend
        
        figure(100);
        show(map);
        hold on
        for j = 1:1:size(locations,1)
            if i <= sizes(j)
                if size(locations{j},1) == 1
                   plot(locations{j}(i,1),locations{j}(i,2), 'ko');
                else
                   plot(locations{j}(i,1),locations{j}(i,2), 'ro');
                end
            else
                if size(locations{j},1) == 1
                   plot(locations{j}(sizes(j),1),locations{j}(sizes(j),2), 'ko');
                else
                   plot(locations{j}(sizes(j),1),locations{j}(sizes(j),2), 'ro');
                end
            end
        end
        hold off
        set(gcf,'color','white');
        pause(0.001);
        
    end
    bot_locs = zeros(size(locations,1),2);
    for j = 1:1:size(locations,1)
        if size(locations{j},1) <= Tend
            bot_locs(j,:) = locations{j}(size(locations{j},1),1:2);
        else
            bot_locs(j,:) = locations{j}(Tend,1:2);
       end
    end
end

%number priority calculation
function number_priority = f(n,nT)
number_priority = n/nT;
end
%distance priority calculation
function distance_priority = d(x1,y1,x2,y2)
dist = abs(x1-x2) + abs(y1-y2);
beta = 4;
distance_priority = beta*(1. / (1 + exp(dist/100)));
end
%total priority of a zone
function zone_priority = p(n,nT,x1,y1,x2,y2)
zone_priority =  f(n,nT) + d(x1,y1,x2,y2);
end


function [max_elements, col_indices] = find_max_priority(matrix)
    % find_max_in_rows finds the largest element in each row of a matrix
    % and the corresponding column index.
    %
    % Syntax: [max_elements, col_indices] = find_max_priority(matrix)
    %
    % Input:
    %   matrix: An n x n matrix
    %
    % Outputs:
    %   max_elements: A row vector containing the largest element in each row
    %   col_indices: A row vector containing the column index of the largest
    %                element in each row
    
    [n, ~] = size(matrix);  % Get the number of rows
    max_elements = zeros(1, n);  % Initialize output vectors
    col_indices = zeros(1, n);
    
    for i = 1:n
        [max_elements(i), col_indices(i)] = max(matrix(i, :));
    end

end


%Robot task allotment in each zone
%cost function
function [c, new_charge_level] = cost(x1,y1,x2,y2,x3,y3,charge_level)
    max_charge = 100;
    max_distance = 2000;
    k = max_charge/max_distance; %Distance travelled by robot to lose all its charge.(in m)
    alpha = 0.1;
    dist = abs(x1-x2) + abs(y1-y2);
    dist2 = 2*(abs(x3-x2) + abs(y3-y2));
    c = alpha*(dist+dist2) - (1-alpha)*(charge_level- ((dist+dist2)*k));
    new_charge_level = charge_level - ((dist+dist2)*k);
end

function [min_val, row_num, col_num] = findMinAndIndices(A)
%FINDMINANDINDICES Finds the minimum value and its row and column indices
%   [min_val, row_num, col_num] = findMinAndIndices(A) returns the minimum
%   value in the matrix A, along with its row number and column number.

    [min_val, index] = min(A(:)); % Find the minimum value and its linear index
    
    % Convert linear index to row and column indices
    [row_num, col_num] = ind2sub(size(A), index);
end



function [map,poses] = Path_Planner(start, stop)
        x = 100;
        y = 80;
        map = binaryOccupancyMap(x,y,1);
        occ = zeros(80,100);
        occ(1,:) = 1;
        occ(end,:) = 1;
        occ(:,1) = 1;
        occ(:,end) = 1;

        num_pick_stations = 3;
        j = 1;
        for i = 1:num_pick_stations
            occ(1:5,5*j) = 1;
            occ(1:5,5*j + 5) = 1;
            j = j+3;
        end

        num_rep_stations = 3;
        q = 1;
        for i = 1:num_rep_stations
            occ(y - 5*q,1:5) = 1;
            occ(y - 5*q -5,1:5) = 1;
            q = q+3;
        end

        %num_charge_stations = 3;
        %z = 1;
        %for i = 1:num_charge_stations
            %occ(y - 5*z,x-4:x) = 1;
            %occ(y - 5*z -5,x-4:x) = 1;
            %z = z+3;
        %end

        num_x_stacks = 16;
        num_y_stacks = 4;
        for k = 0:num_y_stacks-1
            for i = 0:num_x_stacks-1
                for j = 0:5
                    occ(15 + 2*j + 13*k, 10 + 5*i) = 1;
                    occ(15 + 2*j + 13*k, 10 + 5*i + 2) = 1;
                end
            end
        end

        %for i = 0:9
            %occ(78,7 + 8*i) = 1;
            %occ(76,7 + 8*i) = 1;
        %end

        %for i = 0:4
            %occ(74,27 + 5*i) = 1;
        %end

        setOccupancy(map, occ)

        %figure
        %show(map)
        %title('Warehouse Floor Plan')


        % Create map that will be updated with sensor readings
        estMap = occupancyMap(occupancyMatrix(map));

        % Create a map that will inflate the estimate for planning
        inflateMap = occupancyMap(estMap);

        vMap = validatorOccupancyMap;
        vMap.Map = inflateMap;
        vMap.ValidationDistance = .1;
        planner = plannerHybridAStar(vMap,"MinTurningRadius",3);

        entrance = start;
        packagePickupLocation = stop;
        route = plan(planner, entrance, packagePickupLocation);
        route = route.States;

        % Get poses from the route.
        rsConn = reedsSheppConnection('MinTurningRadius', planner.MinTurningRadius);
        startPoses = route(1:end-1,:);
        endPoses = route(2:end,:);
        %fprintf("Sp = %d, ep = %d", size(startPoses,1),size(endPoses,1));
        rsPathSegs = connect(rsConn, startPoses, endPoses);
        poses = [];
        for i = 1:numel(rsPathSegs)
            lengths = 0:0.1:rsPathSegs{i}.Length;
            [pose, ~] = interpolate(rsPathSegs{i}, lengths);
            poses = [poses; pose];
        end

        %figure
        %show(planner)
        %title('Initial Route to Package')

         
end

function m = FindMaxTimeStep(locations)
    m =1;
    for i = 1:1:size(locations,1)
        if size(locations{i},1) >= m
            m = size(locations{i},1);
        end
    end
end

function tasks = checkOcc(tasks)
%Initializing pod locations according to warehouse plan.
    p_locs = zeros(768,2);%PX2 matrix
    x_stacks = 16;%number of stacks of pods in the x direction
    y_stacks = 4;%number of stacks of pods in the y direction
    z = 1;%used for filling pod locations
    while(z<=1)
        for k = 0:y_stacks-1
           for i = 0:x_stacks-1
              for j = 0:5
                  p_locs(z,:) = [10+5*i, 15 + 2*j + 13*k];
                  p_locs(z+1,:) = [10+5*i+2, 15 + 2*j + 13*k];
                  z = z+2;
              end
           end
        end
    end
    for i = 1:1:size(tasks,1)
        for j = 1:1:size(p_locs,1)
            if isequal(tasks(i,:),p_locs(j,:))
                tasks(i,:) = [p_locs(j,1)+1,p_locs(j,2)];
                break
            end
        end
    end
end

function Robot_distances = dCal(locations, bot_locs)
    Robot_distances = zeros(size(bot_locs,1),1);
    for i = 1:1:size(locations,1)
        if size(locations{i},1) ~= 1
            A = locations{i}(:,1:2);
            B = A;
            B(1,:) = [];
            C = A;
            C = C(1:end-1,:);
            D = C-B;
            D = D.^2;
            E = D(:,1) + D(:,2);
            F = sqrt(E);
            G = sum(F,1);
            Robot_distances(i,1) = G;
        else
            Robot_distances(i,1) = 0;
        end
    end
end
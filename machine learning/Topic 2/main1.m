clear all; close all; clc;

N = 100;
M = 1000;
test_points = zeros(M,3);

x0 = rand()*2-1;
y0 = rand()*2-1;
x1 = rand()*2-1;
y1 = rand()*2-1;

g0 = x0*y1 - x1*y0;
g1 = y0-y1;
g2 = x1-x0;
g = [g0;g1;g2];
w = [1;1;1];
input = rand(N,2)*2-1;
in_2 = [ones(N,1),input];

flag = 1;
step_count = 0;
while(flag == 1)

    plot([-1,1,1,-1,-1],[-1,-1,1,1,-1]);

    hold on;
    scatter(input(:,1),input(:,2));
    plot([-1,1],[(-g0+g1)/g2,(-g0-g1)/g2],'b');
    plot([-1,1],[(-w(1)+w(2))/w(3),(-w(1)-w(2))/w(3)],'r');
    legend('Ground Truth','PLA output');
    axis equal;
    
    g_out = sign(in_2 * g);
    w_out = sign(in_2 * w);
    
    I = find(g_out~=w_out);
    if(isempty(I))
        flag = 0;
    else
        h = I(unidrnd(length(I)));
        w = w + g_out(h)*[1;input(h,1);input(h,2)];
    end
    step_count = step_count + 1;
    
    
end
Count = step_count;
test_points(:,1) = 1;
test_points(:,2:3) = rand(M,2)*2-1;
Prob = length(find(sign(test_points*g)~=sign(test_points*w)))/M;

disp(step_count);
disp(Prob);

    
plot([-1,1,1,-1,-1],[-1,-1,1,1,-1]);
hold on;
scatter(input(:,1),input(:,2));
plot([-1,1],[(-g0+g1)/g2,(-g0-g1)/g2],'b');
plot([-1,1],[(-w(1)+w(2))/w(3),(-w(1)-w(2))/w(3)],'r');
legend('Ground Truth','PLA output');
axis equal;
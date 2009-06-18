function bench_figure2d( x, y, z, frustums, bvolumestr, frustumstr )
% y axes des bounding volumes
% x axes des frutums
% z durées
% frustums : nombre fixe de frustums
t = z( :, find(x == frustums));
t = t( :, 1);

% Create figure
figure1 = figure;

% Create axes
axes('Parent',figure1,'YScale','log','YMinorTick','on','YMinorGrid','on',...
    'XScale','log',...
    'XMinorTick','on',...
    'XMinorGrid','on');
box('on');
grid('on');
hold('all');

% Create loglog
loglog(y,t');

% Create xlabel
xlabel(bvolumestr);

% Create ylabel
ylabel('Duration in sec');

% Create title
title([ 'Frustum culling with ', num2str( frustums ), ' ', frustumstr]);

end


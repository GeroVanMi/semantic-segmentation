rsync --progress -aP ./ s_01hsv9an0e4xr06z1daq0kqx1v@ssh.lightning.ai:/teamspace/studios/this_studio/
while inotifywait -r -e modify,create,delete ./; do
    rsync --progress -aP ./ s_01hsv9an0e4xr06z1daq0kqx1v@ssh.lightning.ai:/teamspace/studios/this_studio/
done

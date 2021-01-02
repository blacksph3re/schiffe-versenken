#!/bin/sh

rsync_cmd()
{
  rsync -av --exclude="data" .. taurus:/home/s5265404/diplomarbeit
}

rsync_cmd

while inotifywait -r -e close_write,modify,move,attrib,delete,create ..
do
	echo "updating"
  rsync_cmd
done

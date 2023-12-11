#!/bin/bash
UID=$(id -u)
GID=$(id -g)
UNAME=$(whoami)
docker exec -it --user $UID:$GID  bevfusion_$UNAME bash


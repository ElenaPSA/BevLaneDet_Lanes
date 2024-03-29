FROM nvcr.io/nvidia/pytorch:22.11-py3

# install packages
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends git vim libgtk2.0-dev tmux libmagickwand-dev ffmpeg libsm6 libxext6 && \
    rm -rf /var/cache/apk/*

# Remove the following section if SSH connection is not used
##########################################################
#################SSH and LDAP Configuration###############
##########################################################
RUN apt-get update && apt-get install -y --no-install-recommends supervisor \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/log/supervisor
RUN mkdir /var/run/nscd
COPY files/bev_lane_det/docker/files/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN apt-get update && apt-get install -y --no-install-recommends openssh-server libpam-ldap libnss-ldap ldap-utils nscd sudo cifs-utils vim nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# copy custom scripts
RUN mkdir -p /opt/bin/probayes
COPY files/bev_lane_det/docker/files/fetch_ssh_public_keys.sh /opt/bin/probayes/fetch_ssh_public_keys.sh

RUN chmod a+x /opt/bin/probayes/fetch_ssh_public_keys.sh

# create dir for cache
RUN mkdir -p /var/cache/ldap4ssh

# copy and set config
COPY files/bev_lane_det/docker/files/libnss-ldap.conf /etc/libnss-ldap.conf
COPY files/bev_lane_det/docker/files/pam_ldap.conf /etc/pam_ldap.conf
COPY files/bev_lane_det/docker/files/ldap.conf /etc/ldap.conf
COPY files/bev_lane_det/docker/files/pam_mkhomedir.conf /usr/share/pam-configs/my_mkhomedir
RUN sed -i 's/^#AuthorizedKeysCommand .*$/AuthorizedKeysCommand \/opt\/bin\/probayes\/fetch_ssh_public_keys.sh/' /etc/ssh/sshd_config
RUN sed -i 's/^#AuthorizedKeysCommandUser .*$/AuthorizedKeysCommandUser root/' /etc/ssh/sshd_config
RUN sed -i 's/^#PasswordAuthentication .*$/PasswordAuthentication no/' /etc/ssh/sshd_config
RUN sed -E -i 's/^(passwd|group|shadow):.*$/& ldap/' /etc/nsswitch.conf
RUN mkdir -p /etc/ldap
RUN echo 'TLS_CACERT  /etc/ssl/certs/ca-certificates.crt' >> /etc/ldap/ldap.conf
RUN echo '%probayes   ALL=(ALL:ALL) ALL' >> /etc/sudoers.d/ldap
RUN pam-auth-update --package

# authorize group
RUN echo "ALLOWED_GROUPS='probayes'" >> /etc/default/ldap_authentication.conf

# start service ssh
RUN mkdir /var/run/sshd

COPY files/bev_lane_det/docker/files/config_git_user.sh /opt/bin/probayes/config_git_user.sh
RUN echo 'bash /opt/bin/probayes/config_git_user.sh $USER' >> /etc/skel/.bashrc
RUN echo 'export PYTHONPATH=/workspace:$PYTHONPATH' >> /etc/skel/.bashrc
##########################################################
##########################################################
##########################################################

# Add Python path of main folder and submodule folder to correctly import packages
# that are in submodules. Necessary because submodule is a project directory, not a Python package
# RUN echo 'export PYTHONPATH="${PYTHONPATH}:/workspace/object_detection:/workspace/object_detection/lb_yolox"' >> /etc/skel/.bashrc

# Expose port for the ssh connection
EXPOSE 22
EXPOSE 8888

##########################################################

# Install Python dependencies
COPY requirements_lanes.txt /tmp
RUN pip install --upgrade pip wheel setuptools
RUN pip --no-cache-dir install -r /tmp/requirements_lanes.txt
# Delete the version of opencv from the base image and the one installed by albumentations
RUN pip uninstall -y opencv opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless cv2
RUN rm -rf /usr/local/lib/python3.8/dist-packages/cv2/
# Install the proper version of OpenCV
RUN pip install opencv-contrib-python==4.8.0.74 opencv-contrib-python-headless==4.8.0.74 opencv-python==4.8.0.74

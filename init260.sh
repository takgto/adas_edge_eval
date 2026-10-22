ifconfig eth0 192.168.1.100 netmask 255.255.255.0 up
alias yolov3='cd /home/root/Vitis-AI/examples/VART/yolov3'
alias models='cd /usr/share/vitis_ai_library/models'
alias yolov3s='cd /home/root/Vitis-AI/examples/Vitis-AI-Library/samples/yolov3'
alias model='cd /usr/share/vitis_ai_library/models'
alias cp='cp -i'
alias mv='mv -i'
alias rm='rm -i'
alias ls='ls --color=auto'
alias ls='ls -G'
alias ll='ls -alF'
alias ll='ls -lh'
alias ll='ls -l'
alias la='ls -A'
alias la='ls -a'
alias l='ls -CF'
alias vi='vi -c "set nobackup" -c "set nowritebackup"'

mkdir -p /home/root/Vitis-AI/examples/VART/yolov3
ln -s /home/root/Vitis-AI/examples/Vitis-AI-Library/samples/yolov3/video /home/root/Vitis-AI/examples/VART/yolov3/video
ln -s /usr/share/vitis_ai_library/models/dpu_yolov3/dpu_yolov3.xmodel /home/root/Vitis-AI/examples/VART/dpu_yolov3

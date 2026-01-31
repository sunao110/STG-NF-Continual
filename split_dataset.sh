#python split_npy.py \
#  --source /home/hdd1/sunao/HumanML3D/new_joints \
#  --train /home/hdd1/sunao/HumanML3D/train \
#  --test /home/hdd1/sunao/HumanML3D/test \

python split_dataset.py \
  --source /home/hdd1/sunao/ACMDM \
  --train /home/hdd1/sunao/ACMDM/train \
  --test /home/hdd1/sunao/ACMDM/test \
  --ratio 0.9

import os

path = "/home/t2-503-4090/QianXi/training_data/openpilot_camera_laneline"
f = open("videos.txt", 'w')
f2 = open("plans.txt", 'w')

for file_name in sorted(os.listdir(path)):
    file_name = os.path.join(path, file_name)
    
    if file_name.endswith('.mp4'):
        f.write(file_name + '\n')
    else:
        f2.write(file_name + '\n')



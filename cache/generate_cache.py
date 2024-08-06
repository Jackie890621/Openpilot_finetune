import os
import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python generate_cache.py <dataset_directory>")
    else:
        dataset_dir = sys.argv[1]

        f = open("videos.txt", 'w')
        f2 = open("plans.txt", 'w')

        for file_name in sorted(os.listdir(dataset_dir)):
            file_name = os.path.join(dataset_dir, file_name)
            
            if file_name.endswith('.mp4'):
                f.write(file_name + '\n')
            else:
                f2.write(file_name + '\n')


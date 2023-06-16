import os
import json
import argparse
import numpy as np

def bb_reformat(bboxin, render_width, render_height):
    tlx, tly, w, h = bboxin

    cx = tlx + w/2
    cy = tly + h/2
    return [cx/render_width, cy/render_height, w/render_width, w/render_height]

def convert_dataset(bop_dir, output_dir, val_frac = 0.1, width=720, height=540):
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    label_dir = os.path.join(output_dir, "labels")
    os.makedirs(label_dir, exist_ok=True)

    train_file = os.path.join(output_dir, "train.txt")
    val_file = os.path.join(output_dir, "validation.txt")

    dirs = next(os.walk(bop_dir))[1]

    # find all classes and count instances
    class_counts = {}
    for dir in dirs:
        # Load the BOP-scenewise dataset annotation file
        poses_and_ids_file = os.path.join(bop_dir, '{}/scene_gt.json'.format(dir))
        with open(poses_and_ids_file, 'r') as f:
            poses_and_ids = json.load(f)
        for scene_id, object_pose_id in poses_and_ids.items():
            for obj_pi in object_pose_id:
                if obj_pi["obj_id"] in class_counts:
                    class_counts[obj_pi["obj_id"]] += 1
                else:
                    class_counts[obj_pi["obj_id"]] = 1
    classes = list(sorted(class_counts.keys()))
    cls2clsind = {} # yolov3 needs 0 indexed indices as class markers
    for i, clc in enumerate(classes):
        cls2clsind[clc] = i

    for n, dir in enumerate(dirs):
        # Load the BOP-scenewise dataset annotation file
        poses_and_ids_file = os.path.join(bop_dir, '{}/scene_gt.json'.format(dir))
        with open(poses_and_ids_file, 'r') as f:
            poses_and_ids = json.load(f)
        bbox_and_vis_file = os.path.join(bop_dir, '{}/scene_gt_info.json'.format(dir))
        with open(bbox_and_vis_file, 'r') as f:
            bbox_and_vis = json.load(f)
        for scene_id, object_pose_id in poses_and_ids.items():
            # Copy the image file to the output directory
            base_filename = str(dir).zfill(6) + "_" + str(scene_id).zfill(6)
            image_filename = base_filename + '.jpg'
            image_path = os.path.join(bop_dir, dir, 'rgb', str(scene_id).zfill(6) + '.jpg')
            image_output_path = os.path.join(img_dir, image_filename)
        
            os.system(f'cp {image_path} {image_output_path}')

            # randomize if to put in val or train
            use_for_val = np.random.rand() < val_frac
            with open(val_file if use_for_val else train_file, 'a') as f:
                f.write(os.path.join("./images/", image_filename) + "\n")

            label_filename = base_filename + '.txt'
            label_path_txt = os.path.join(label_dir, label_filename)
            with open(label_path_txt, "w") as label_file:
                for i, obj_pi in enumerate(object_pose_id):
                    bbox = bbox_and_vis[scene_id][i]["bbox_obj"]
                    if bbox != [-1, -1, -1, -1]:
                        cx, cy, w, h = bb_reformat(bbox, width, height)
                        label_file.write("{} {} {} {} {}\n".format(cls2clsind[obj_pi["obj_id"]], cx, cy, w, h))
        print(f"{n+1} out of {len(dirs)} directories converted")
    print('Conversion completed successfully.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert BOP-scenewise dataset format to Ultralytics YOLOv3 dataset format')
    parser.add_argument('bop_dir', help='Path to the BOP-scenewise dataset directory')
    parser.add_argument('output_dir', help='Output directory for the converted dataset')
    parser.add_argument('val_frac', help='Fraction of data to use for validation', nargs='?', const=1, type=float, default=0.1)
    args = parser.parse_args()

    convert_dataset(args.bop_dir, args.output_dir, args.val_frac)

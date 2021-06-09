DATASET_DIR=/home/ubuntu/576_Project_Colorization/img_data/example/train

python inference_bbox.py --test_img_dir $DATASET_DIR --filter_no_obj

mkdir -p data/coco && cd data/coco
wget -c http://images.cocodataset.org/zips/train2017.zip &
wget -c http://images.cocodataset.org/zips/val2017.zip &
wget -c http://images.cocodataset.org/zips/test2017.zip &

# ==== 2017 Annotations ====
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip &
wget -c http://images.cocodataset.org/annotations/image_info_test2017.zip &

# ==== 2014 Images ====
wget -c http://images.cocodataset.org/zips/train2014.zip &
wget -c http://images.cocodataset.org/zips/val2014.zip &
wget -c http://images.cocodataset.org/zips/test2014.zip &

# ==== 2014 Annotations ====
wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip  &
wget -c http://images.cocodataset.org/annotations/image_info_test2014.zip &

# Wait for all backgrounded wget processes to finish before proceeding
wait

# ==== Extract all ====
unzip '*.zip'
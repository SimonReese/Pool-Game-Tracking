# Results folder structure
This folder contains outputs of the system runned in live and evaluation modes.
Each game clip has already been ran and the system has saved output in the correspondig game clip subfolder  of this results folder.

Each game clip folder contains the following:

- output-video-gameX_clipY.mp4: a video showing the drawing superimposed over the real video,
- bboxes-frame_first.png: bounding boxes of detected balls drawn over the first frame provided in the dataset folder
- bboxes-frame_last.png: bounding boxes of detected balls drawn over the last frame provided in the dataset folder
- masks subfolder: containing the system generated masks on every frame provided in the dataset folder
- bounding_boxes subfolder: containing the system generated bounding boxes file on every frame provided in the dataset folder
- a metrics.txt file containing the obtained results of the system after being run in evaluation mode.

The masks files are consistent with the mask provided in the test dataset. In order to view easily those masks, a simple executable is provided, and can be launched as follow:

```bash
./open_mask path/to/mask/file.png path/to/antoher/mask/file.png
```

usually it is useful to compare true masks vs predicted masks, thus it requires two mask paths.
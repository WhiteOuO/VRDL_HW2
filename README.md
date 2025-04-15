# NYCU VRDL HW2  
StudentID:110550128  
Name:蔡耀霆
## Introduction:

This project tackles an object detection and digit recognition task using a Faster R-CNN model with a ResNet50-FPN backbone. The goal is to accurately detect and recognize sequences of digits in complex images.

To improve training efficiency and address memory constraints, the following techniques were implemented:

- **Pre-saved `.pt` tensors** for faster data loading, avoiding real-time augmentation and reducing computational overhead.
- **Proper data augmentation** proper data augmentations are applied to this digits recognitions tasks to improve model's robustness.
- **Split training strategy**, where the full dataset is divided into multiple subsets, and only one subset is loaded into memory per epoch to alleviate RAM pressure.
- **Learning rate decay with early stopping**, allowing efficient convergence and reducing the risk of overfitting.
- **Post-processing of digit sequence prediction**, where predicted boxes are sorted left-to-right, and corresponding digit labels are concatenated to form the final prediction.

The model outputs both bounding box predictions (`pred.json`) and digit sequence classifications (`pred.csv`), evaluated on the official test set.

## How to Run
### Basic training process & prediction
1.Run the image_aug_to_pt.py and turn all images and corresponding annotations into a pt file.  
2.Run the pt_spilt.py.  
3.Run the main_train_cycle.py.  
4.Run prediction.py.  
Other files are for findings.  
## Performance snapshot
![image](https://github.com/user-attachments/assets/dbff9a04-fd66-4c45-948f-c04daeb32820)

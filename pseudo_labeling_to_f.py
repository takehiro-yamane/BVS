from review import fmeasure, threshold
import segmentation_train_mask_parallel, segmentation_predict

sv_path = '0316/Focal/2image'
pseudo_path = '0505-pseudo'

# threshold.main(sv_path, pseudo_path, 0.8, 0.15)
# segmentation_train_mask_parallel.main_for_together_add(pseudo_path, sv_path)
# segmentation_predict.pred_5groups(pseudo_path)
# fmeasure.main(pseudo_path)

pseudo_path = '0505-pseudo2'
segmentation_predict.pred_5groups(pseudo_path)
fmeasure.main(pseudo_path)
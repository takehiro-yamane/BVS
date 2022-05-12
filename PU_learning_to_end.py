from matplotlib.pyplot import get
import PU_train_0
from review import get_bottom, vis_score, fmeasure, PCA2, PU_pre_prepare
import segmentation_predict
import segmentation_train_mask_parallel
import PU_predict

pu_learning_dir = '0502-pu'
dir_p = '0502-put'
pseudo_p = '0411-pseudo'
lwpath = '0316/Focal/2image'


# PU_train_0.main_for_together(pu_learning_dir)
# print('PU learning finished') 

# PU_pre_prepare.main(pu_learning_dir)
# print('PU pre prepare finished')

# PU_predict.main(pu_learning_dir)
# print('PU predict finished')

# get_bottom.main(dir_p, pu_learning_dir, pseudo_p)
# print('get labels (PU) finished')

# segmentation_train_mask_parallel.main_for_together_add(dir_p, lwpath)
# print('add train finished')

# segmentation_predict.pred_5groups(dir_p)
# print('predict finished')

# fmeasure.main(dir_p)
# print('fmeasure finished')


# ##PU learning confirm
# vis_score.main(pu_learning_dir)
# print('vis-score finished')

PCA2.main(pu_learning_dir)
print('PCA finished')

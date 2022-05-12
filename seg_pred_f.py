import imp
import segmentation_predict
from review import fmeasure

dir_p = '0502-put'
segmentation_predict.pred_5groups(dir_p)
fmeasure.main(dir_p)
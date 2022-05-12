from review import prepare_PU_sv, prepare_PU_oneU, calc_prior
import PU_train_0
from pathlib import Path


parent_path = Path('Result/0415-pu/for_PU_train')
prepare_PU_sv.main(parent_path)
prepare_PU_oneU.main(parent_path)

dir_p = parent_path.parent
pseudo_p = 'Result/0411-pseudo'
calc_prior.main(dir_p, pseudo_p)


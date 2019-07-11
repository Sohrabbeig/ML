from src.dimensionality_reduction_methods import PCA
from src.learners import multiclass_SVM, decision_tree
from src.utils import get_data
import warnings

warnings.filterwarnings("ignore")

X = get_data('final_project/data/train.csv')
low_dimensional_X = PCA(X)

print(low_dimensional_X.shape)

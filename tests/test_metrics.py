import numpy as np
from quantum_ml import metrics

def test_gini_one():
    y_true = [1,1,1,1,1000]
    y_pred = [1,2,3,4,5]
    sample_weight = [1,1,1,1,1]
    
    
    assert np.round(metrics.gini(y_true,y_pred,sample_weight),3) == 0.995 , \
        "sample value should match" 
    
    

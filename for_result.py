import extraction_from_music
import numpy as np 
from tensorflow.keras.models import load_model

########################################### (1,H,W,C) return
def ext_sample_input(file_path):
    sample_data = extraction_from_music.ext_datapoint(file_path)
    sample_data = np.expand_dims(sample_data, axis = 0)
    return sample_data

############################################ 10차원 결과 백터(list) return
def return_result(sample_input, model_path):
    loaded_model = load_model(model_path)
    prediction = loaded_model.predict(sample_input)
    return prediction[0]
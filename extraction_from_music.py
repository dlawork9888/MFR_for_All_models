# (100, 300, 3)의 output
# scaling 없이 => sigmoid로
import librosa
import numpy as np
import librosa.display
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import RegularGridInterpolator #선형보간
import math
from scipy.ndimage import zoom


###################################### Modules

# base y, sr

def ext_base(path):
    y, sr = librosa.load(path)
    return y, sr

# Chroma_stft

def ext_chroma_stft(y, sr):
    chroma_stft = librosa.feature.chroma_stft(y = y, sr = sr)
    # (12,1200)의 형태
    print(f'original chroma_stft.shape : {chroma_stft.shape}')
    print(f'slicing to [:,:1200]')
    chroma_stft = chroma_stft[:,:1200]
    """
    print(f'---------check----------')
    # 모두 양수인지 확인
    if np.all(chroma_stft >= 0):
        print(f'chroma_stft의 모든 원소가 양수')
    else:
        print(f'chroma_stft의 원소 중 음수가 존재')
    # 최대최소값 확인
    print(f'max : {np.max(chroma_stft)}')
    print(f'min : {np.min(chroma_stft)}')
    print(f'------------------------')
    """
    return chroma_stft

# MFCC

def ext_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y = y, sr = sr)
    print(f'original mfcc.shape : {mfcc.shape}')
    print(f'slicing to [:,:1200]')
    mfcc = mfcc[:,:1200]
    """
    print(f'scaling_minmax')
    scaler = MinMaxScaler()
    mfcc = scaler.fit_transform(mfcc.T).T
    print(f'---------check----------')
    if np.all(mfcc >= 0):
        print(f'mfcc의 모든 원소가 양수')
    else:
        print(f'mfcc의 원소 중 음수가 존재')
    print(f'max : {np.max(mfcc)}')
    print(f'min : {np.min(mfcc)}')
    print(f'------------------------')
    """
    
    return mfcc # 2차원

# Tempogram

def ext_tempogram(y, sr):
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    print(f'slicing to [:,:1200]')
    tempogram = tempogram[:,:1200]
    """
    print(f'scaling_minmax')
    scaler = MinMaxScaler()
    tempogram = scaler.fit_transform(tempogram.T).T
    print(f'---------check----------')
    if np.all(tempogram >= 0):
        print(f'mfcc의 모든 원소가 양수')
    else:
        print(f'mfcc의 원소 중 음수가 존재')
    print(f'max : {np.max(tempogram)}')
    print(f'min : {np.min(tempogram)}')
    print(f'------------------------')
    """
    return tempogram # 2차원
 
# 선형 보간
"""
def interpolate_array_height(arr, new_height):
    # 주어진 배열의 크기
    original_height, original_width = arr.shape

    # 새로운 y 값 생성
    y_new = np.linspace(0, original_height - 1, new_height)

    # RegularGridInterpolator를 이용하여 보간된 함수 생성
    interp_func = RegularGridInterpolator((np.arange(original_height), np.arange(original_width)), arr)

    # 보간된 배열 생성
    x_new = np.arange(original_width)
    xx, yy = np.meshgrid(x_new, y_new)
    points = np.array([yy, xx]).transpose((1, 2, 0))
    arr_interpolated = interp_func(points)

    return arr_interpolated
"""

# zoom을 이용한 resizing
def return_zoomed(arr, new_height, new_width):
    zoomed = zoom(arr, (new_height/arr.shape[0], new_width/arr.shape[1]))
    return zoomed


############################################## integrate
# input => 파일 경로
# output => (400, 1200, 3)의 데이터 포인트

def ext_datapoint(path):
    y ,sr = ext_base(path)
    
    feature_list= []
    
    chroma_stft = ext_chroma_stft(y, sr)
    feature_list.append(chroma_stft)
    mfcc = ext_mfcc(y, sr)
    feature_list.append(mfcc)
    tempogram = ext_tempogram(y, sr)
    feature_list.append(tempogram)
    
    # 선형 보간 => zoom 방식
    new_height, new_width = 100,300
    for idx, feature in enumerate(feature_list):
        feature_list[idx] = return_zoomed(feature, new_height, new_width)
        feature_list[idx] = np.expand_dims(feature_list[idx], axis = -1)
    
    print(f'-------------linear interpolating done--------------')
    print(f'-------------adding a dimension done--------------')
    print(f'check!')
    for x in feature_list:
        print(x.shape, end = ' ')
    print()
    
    ### concatenate
    concatenated = np.concatenate(feature_list, axis = -1)
    print(f'-------------concatenating done--------------')
    print(f'check!')
    print(f'concatenated.shape : {concatenated.shape}')

    return concatenated

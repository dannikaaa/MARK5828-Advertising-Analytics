from PIL import Image
import urllib
import requests
from scipy.linalg.blas import ssyrk
from scipy.linalg.lapack import spotrf, sposv
from itertools import product
import pandas as pd
import numpy as np

def variance_inflation_factor(X):
    """
    Calculates the variance inflation factor for each column in a dataframe. Credit to Daniel-Han Chan
    Input:
        X: Dataframe
    Output:
        Dataframe of variance inflation values for each column.
    """
    isDataframe = type(X) is pd.DataFrame
    if isDataframe:
        columns = X.columns
        X = X.values
    n, p = X.shape

    swap = np.arange(p)
    np.random.shuffle(swap)

    XTX = X.T @ X
    XTX = XTX[swap][:,swap]

    select = np.ones(p, dtype = bool)

    temp = XTX.copy().T
    error = 1
    largest = XTX.diagonal().max() // 2
    add = largest
    maximum = np.finfo(np.float32).max

    while error != 0:
        C, error = spotrf(a = temp)
        if error != 0:
            error -= 1
            select[error] = False
            temp[error, error] += add
            error += 1

            add += np.random.randint(1,30)
            add *= np.random.randint(30,50)
        if add > maximum:
            add = largest

    VIF = np.empty(p, dtype = np.float32)
    means = np.mean(X, axis = 0)[swap]

    for i in range(p):
        curr = select.copy()
        s = swap[i]

        if curr[i] == False:
            VIF[s] = np.inf
            continue
        curr[i] = False

        XX = XTX[curr]
        xtx = XX[:, curr]
        xty = XX[:,i]
        y_x = X[:,s]

        theta_x = sposv(xtx, xty)[1]
        y_hat = X[:,swap[curr]] @ theta_x

        SS_res = y_x-y_hat
        SS_res = np.einsum('i,i', SS_res, SS_res)
        #SS_res = np.sum((y_x - y_hat)**2)

        SS_tot = y_x - means[i]
        SS_tot = np.einsum('i,i', SS_tot, SS_tot)
        #SS_tot = np.sum((y_x - np.mean(y_x))**2)
        if SS_tot == 0:
            R2 = 1
            VIF[s] = np.inf
        else:
            R2 = 1 - (SS_res/SS_tot)
            VIF[s] = 1/(1-R2)
        del XX, xtx, xty, y_x, theta_x, y_hat
    if isDataframe:
        df_vif = pd.DataFrame({"vif": VIF})
        df_vif = df_vif.set_index(columns)
        return df_vif
    return VIF

class MSFaceAPIClient():
    def __init__(self, subscription_key, location="westcentralus"):
        self._subscription_key = subscription_key
        self._location = location

    def face_detect_local(self, image_path):
        headers = {
            # Request headers
            'Content-Type': 'application/octet-stream',
            'Ocp-Apim-Subscription-Key': self._subscription_key,
        }

        params = urllib.parse.urlencode({
            # Request parameters
            'returnFaceId': 'true',
            'returnFaceLandmarks': 'true',
            'returnFaceAttributes': "age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise"
        })

        try:
            url = "https://" + self._location + ".api.cognitive.microsoft.com/face/v1.0/detect"
            return requests.post(url, params=params, headers=headers, data=open(image_path, 'rb')).json()
        except Exception as e:
            print("Detect Face Error: " + str(e))
            return None

def get_img_dim(image_name):
    im = Image.open(image_name)
    return im.size

def get_face_loc(pic_size, face_pos, face_size, str_out):
    """Can use any dimension"""

    face_center = (face_pos + face_pos + face_size) / 2

    pic_divider = pic_size / 3

    for i in range(3):
        if face_center < pic_divider * (i + 1):
            return str_out[i]
    return None

def face_json_to_df(data, image_name):
    # Selfie: Use the biggest rectangle as main data

    width, height = get_img_dim(image_name)

    data[0]["num_faces"] = len(data)
    data[0]["pic_width"] = width
    data[0]["pic_height"] = height

    # Hard-coded Controlled Variables
    data[0]["user_num_followers"] = 100
    data[0]["user_num_following"] = 100
    data[0]["user_num_posts"] = 50
    data[0]["year_uploaded"] = 2019
    data[0]["month_uploaded"] = 2
    data[0]["day_uploaded"] = 25


    accessories = {
        "glasses": 0.0,
        "headwear": 0.0,
        "mask": 0.0
    }
    try:
        for item in data[0]["faceAttributes"]["accessories"]:
            accessories[item["type"]] = item["confidence"]

    except Exception as e:
        pass

    data[0]["faceAttributes"]["accessories"] = accessories

    hair_colors = {
        "black": 0.0,
        "blond": 0.0,
        "brown": 0.0,
        "gray": 0.0,
        "other": 0.0,
        "red": 0.0
    }

    try:
        for item in data[0]["faceAttributes"]["hair"]["hairColor"]:
            hair_colors[item["color"]] = item["confidence"]

    except Exception as e:
        pass

    data[0]["faceAttributes"]["hair"]["hairColor"] = hair_colors

    df_face = pd.io.json.json_normalize(data, sep="_")

    rename_cols = {
        "faceAttributes_accessories_glasses": "has_glasses",
        "faceAttributes_accessories_headwear": "has_headwear",
        "faceAttributes_accessories_mask": "has_mask",
        "faceAttributes_age": "age",
        "faceAttributes_blur_blurLevel": "blur_level",
        "faceAttributes_emotion_anger": "feels_angry",
        "faceAttributes_emotion_contempt": "feels_contempt",
        "faceAttributes_emotion_disgust": "feels_digust",
        "faceAttributes_emotion_fear": "feels_fear",
        "faceAttributes_emotion_happiness": "feels_happy",
        "faceAttributes_emotion_neutral": "feels_neutral",
        "faceAttributes_emotion_sadness": "feels_sad",
        "faceAttributes_emotion_surprise": "feels_surprise",
        "faceAttributes_exposure_exposureLevel": "exposure_level",
        "faceAttributes_facialHair_beard": "has_beard",
        "faceAttributes_facialHair_moustache": "has_moustache",
        "faceAttributes_facialHair_sideburns": "has_sideburns",
        "faceAttributes_gender": "gender",
        "faceAttributes_glasses": "glasses_type",
        "faceAttributes_hair_bald": "is_bald",
        "faceAttributes_hair_hairColor_black": "has_black_hair",
        "faceAttributes_hair_hairColor_blond": "has_blonde_hair",
        "faceAttributes_hair_hairColor_brown": "has_brown_hair",
        "faceAttributes_hair_hairColor_gray": "has_gray_hair",
        "faceAttributes_hair_hairColor_other": "has_other_hair",
        "faceAttributes_hair_hairColor_red": "has_red_hair",
        "faceAttributes_headPose_pitch": "head_pitch",
        "faceAttributes_headPose_roll": "head_roll",
        "faceAttributes_headPose_yaw": "head_yaw",
        "faceAttributes_makeup_eyeMakeup": "has_eye_makeup",
        "faceAttributes_makeup_lipMakeup": "has_lip_makeup",
        "faceAttributes_noise_noiseLevel": "noise_level",
        "faceAttributes_smile": "is_smiling",
        "faceRectangle_left": "face_rectangle_left",
        "faceRectangle_top": "face_rectangle_top",
        "faceRectangle_width": "face_rectangle_width",
        "faceRectangle_height": "face_rectangle_height"
    }

    df_face = df_face.rename(columns=rename_cols)

    x_vars = [
    "pic_width",
    "face_rectangle_left",
    "face_rectangle_width"
    ]

    x_keys = [
        "left",
        "center",
        "right"
    ]

    y_vars = [
        "pic_height",
        "face_rectangle_width",
        "face_rectangle_height"
    ]

    y_keys = [
        "top",
        "center",
        "bottom"
    ]

    x_y_keys = list(product(y_keys, x_keys))
    x_y_values = ["_".join(tup) for tup in x_y_keys]
    x_y_map = dict(zip(x_y_keys, x_y_values))

    df_face["face_x"] = df_face[x_vars].apply(lambda x: get_face_loc(x[0], x[1], x[2], x_keys), axis=1)
    df_face["face_y"] = df_face[y_vars].apply(lambda x: get_face_loc(x[0], x[1], x[2], y_keys), axis=1)
    df_face["face_pos"] = df_face[["face_y", "face_x"]].apply(lambda x: x_y_map[(x[0], x[1])], axis=1)

    drop_cols = [
        "faceAttributes_hair_invisible",             # Multicollinearity with hair_color columns
        "faceAttributes_exposure_value",             # Multicollinearity with exposure level
        "faceAttributes_noise_value",                # Multicollinearity with noise level
        "faceAttributes_blur_value",                 # Multicollinearity with blur level
        "faceAttributes_occlusion_eyeOccluded",      # Remove cause i dont know what this means
        "faceAttributes_occlusion_foreheadOccluded",
        "faceAttributes_occlusion_mouthOccluded",
        "face_x",
        "face_y",
        "faceId"
    ]

    df_face = df_face.drop(columns=drop_cols)

    drop_cols = [
        "faceAttributes_accessories",
        "faceAttributes_hair_hairColor",
    ]

    try:
        df_face = df_face.drop(columns=drop_cols)
    except Exception as e:
        pass

    df_face_columns_filter = list(df_face.columns)
    df_face_columns_filter = list(filter(lambda x: "faceLandmarks_" not in x, df_face_columns_filter))
    df_face = df_face.loc[:, df_face_columns_filter]
    return df_face




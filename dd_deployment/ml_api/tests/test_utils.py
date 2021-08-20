from __future__ import absolute_import

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from birnna_model import pipeline
from birnna_model.predict import make_prediction

from api.utils import get_top3_index, get_bucket_object, search_key_word_index, generate_keyword_objects


def test_get_top3_index():
    # Given
    prob_array = np.array([0.2, 0.5, 0.1, 0.15, 0.25])

    # When
    top3_index = get_top3_index(prob_array)

    # Then
    assert top3_index == [1, 4, 0]


def test_get_bucket_object():
    # Given
    prob_array = [0.2, 0.5, 0.1, 0.15, 0.25]
    target_dict = {0: ("Unknown", "subsys", "", "Unknown"),
                   1: ("Test_Plan", "subsys", "", "Test_Plan"),
                   2: ("Intel dtt Driver", "subsys", "Intel dtt Driver"),
                   3: ("touchpad driver", "compPart", "touchpad, touchpad abc", "Driver"),
                   4: ("bios - system bios", "compPart", "bios,system bios", "BIOS,BIOS X")}
    index1 = 1
    index2 = 4

    # Then
    assert get_bucket_object(prob_array, index1, target_dict) == {"type": "subsys", "value": "Test_Plan",
                                                                  "valueSubsys": ["Test_Plan"],
                                                                  "probability": 0.5,
                                                                  "realCompPartNames": None}
    assert get_bucket_object(prob_array, index2, target_dict) == {"type": "compPart", "value": "bios - system bios",
                                                                  "valueSubsys": ["BIOS", "BIOS X"],
                                                                  "probability": 0.25,
                                                                  "realCompPartNames": ["bios", "system bios"]}


def test_search_key_word_index():
    # Given
    key_word = ["device", "win", "testing"]
    pred_weight = ["0.23", "0.4", "0.37"]
    input_text = "XSEF02381 device:0.8 win.8 XKSDAi081"

    # When
    key_word, key_word_index, pred_weight = search_key_word_index(key_word, pred_weight, input_text)

    # Then
    assert key_word == ["device", "win"]
    assert key_word_index == [10, 21]
    assert pred_weight == ["0.23", "0.4"]


def test_generate_keyword_objects(test_input_data):
    # Given
    df_raw = pd.DataFrame({'short_desc': [test_input_data.get('ShortDesc')],
                           'long_desc': [test_input_data.get('LongDesc')],
                           'steps_to_repro': [test_input_data.get('StepsToRepro')],
                           'frequency': [test_input_data.get('Frequency')],
                           'business_segment': [test_input_data.get('BizSegId')]})
    preporcessing_pipe = Pipeline(pipeline.obs_pipe.steps[0:2])
    df_transformed = preporcessing_pipe.transform(df_raw)

    subject = make_prediction(input_data=df_raw)
    tot_pred_weights = subject['predictions'][1][0].tolist()

    # When
    key_word_objects = generate_keyword_objects(df_raw, df_transformed, tot_pred_weights)

    # Then
    for _, key_word_object in key_word_objects.items():
        assert len(key_word_object['keyWord']) == len(key_word_object['keyWordStartIndex'])
        assert len(key_word_object['keyWord']) == len(key_word_object['weight'])

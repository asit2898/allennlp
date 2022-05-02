from typing import Any, Dict, List, Optional, Union, cast
from overrides import overrides

import os
import random
import pickle
import logging

import numpy as np
import pyarrow as pa

from dataclasses import dataclass

from functools import lru_cache

from torch._C import dtype

from allennlp.data.fields import Field, TextField, LabelField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, PretrainedTransformerTokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.dataset_readers.huggingface import HuggingFaceDatasetReader


from datasets import load_dataset
from datasets.features import Features, Value, ClassLabel, Sequence
from datasets.arrow_dataset import Dataset
from datasets import set_caching_enabled

logger = logging.getLogger(__name__)

set_caching_enabled(os.environ.get('HF_DISABLE_CACHE', '').strip() != "1")


@dataclass
class SentencePairFeature:
    key1: str
    key2: str


@DatasetReader.register("nlu")
class NLUDatasetReader(HuggingFaceDatasetReader):
    def __init__(
        self,
        *,
        path: Optional[str] = None,
        name: Optional[str] = None,
        tokenizer: Tokenizer,
        token_indexers: Dict[str, TokenIndexer] = None,
        combine_input_fields: Optional[bool] = None,
        ensure_whitespace_between: bool = False,
        combine_opposite: bool = False,
        config_kwargs: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        super(HuggingFaceDatasetReader,self).__init__(manual_distributed_sharding=False, **kwargs)  # Right now disabled
        self.class_mappings = {'all_scenarios':{0: 'alarm_query', 1: 'alarm_remove', 2: 'alarm_set', 3: 'audio_volume_down', 4: 'audio_volume_mute', 5: 'audio_volume_other', 6: 'audio_volume_up', 7: 'calendar_query', 8: 'calendar_remove', 9: 'calendar_set', 10: 'cooking_query', 11: 'cooking_recipe', 12: 'datetime_convert', 13: 'datetime_query', 14: 'email_addcontact', 15: 'email_query', 16: 'email_querycontact', 17: 'email_sendemail', 18: 'general_affirm', 19: 'general_commandstop', 20: 'general_confirm', 21: 'general_dontcare', 22: 'general_explain', 23: 'general_greet', 24: 'general_joke', 25: 'general_negate', 26: 'general_praise', 27: 'general_quirky', 28: 'general_repeat', 29: 'iot_cleaning', 30: 'iot_coffee', 31: 'iot_hue_lightchange', 32: 'iot_hue_lightdim', 33: 'iot_hue_lightoff', 34: 'iot_hue_lighton', 35: 'iot_hue_lightup', 36: 'iot_wemo_off', 37: 'iot_wemo_on', 38: 'lists_createoradd', 39: 'lists_query', 40: 'lists_remove', 41: 'music_dislikeness', 42: 'music_likeness', 43: 'music_query', 44: 'music_settings', 45: 'news_query', 46: 'play_audiobook', 47: 'play_game', 48: 'play_music', 49: 'play_podcasts', 50: 'play_radio', 51: 'qa_currency', 52: 'qa_definition', 53: 'qa_factoid', 54: 'qa_maths', 55: 'qa_stock', 56: 'recommendation_events', 57: 'recommendation_locations', 58: 'recommendation_movies', 59: 'social_post', 60: 'social_query', 61: 'takeaway_order', 62: 'takeaway_query', 63: 'transport_query', 64: 'transport_taxi', 65: 'transport_ticket', 66: 'transport_traffic', 67: 'weather_query'},'cooking': {0: 'cooking_query', 1: 'cooking_recipe'}, 'transport': {0: 'transport_query', 1: 'transport_taxi', 2: 'transport_ticket', 3: 'transport_traffic'}, 'email': {0: 'email_addcontact', 1: 'email_query', 2: 'email_querycontact', 3: 'email_sendemail'}, 'general': {0: 'general_affirm', 1: 'general_commandstop', 2: 'general_confirm', 3: 'general_dontcare', 4: 'general_explain', 5: 'general_greet', 6: 'general_joke', 7: 'general_negate', 8: 'general_praise', 9: 'general_quirky', 10: 'general_repeat'}, 'qa': {0: 'qa_currency', 1: 'qa_definition', 2: 'qa_factoid', 3: 'qa_maths', 4: 'qa_stock'}, 'recommendation': {0: 'recommendation_events', 1: 'recommendation_locations', 2: 'recommendation_movies'}, 'audio': {0: 'audio_volume_down', 1: 'audio_volume_mute', 2: 'audio_volume_other', 3: 'audio_volume_up'}, 'alarm': {0: 'alarm_query', 1: 'alarm_remove', 2: 'alarm_set'}, 'social': {0: 'social_post', 1: 'social_query'}, 'datetime': {0: 'datetime_convert', 1: 'datetime_query'}, 'iot': {0: 'iot_cleaning', 1: 'iot_coffee', 2: 'iot_hue_lightchange', 3: 'iot_hue_lightdim', 4: 'iot_hue_lightoff', 5: 'iot_hue_lighton', 6: 'iot_hue_lightup', 7: 'iot_wemo_off', 8: 'iot_wemo_on'}, 'lists': {0: 'lists_createoradd', 1: 'lists_query', 2: 'lists_remove'}, 'calendar': {0: 'calendar_query', 1: 'calendar_remove', 2: 'calendar_set'}, 'music': {0: 'music_dislikeness', 1: 'music_likeness', 2: 'music_query', 3: 'music_settings'}, 'play': {0: 'play_audiobook', 1: 'play_game', 2: 'play_music', 3: 'play_podcasts', 4: 'play_radio'}, 'takeaway': {0: 'takeaway_order', 1: 'takeaway_query'}}
        self.scenario = path.split('/')[-1]
        self._tokenizer = tokenizer
        if isinstance(self._tokenizer, PretrainedTransformerTokenizer):
            assert not self._tokenizer._add_special_tokens
            # TODO what if no combining is required?
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if combine_input_fields is not None:
            self._combine_input_fields = combine_input_fields
        else:
            self._combine_input_fields = isinstance(self._tokenizer, PretrainedTransformerTokenizer)

        self._ensure_whitespace_between = ensure_whitespace_between
        if ensure_whitespace_between:
            assert self._combine_input_fields

        self.path = path
        self.name = name

        if config_kwargs is None:
            config_kwargs = {}
        self.config_kwargs = config_kwargs


        self._keys_to_combine = [
            ("sentence1", "sentence2"),
            ("premise", "hypothesis"),
            ("question", "sentence"),
            ("question1", "question2"),
            ("question", "passage"),
        ]
        self._combine_opposite = combine_opposite
        if combine_opposite:
            self._keys_to_combine = [
                (second, first)
                for first, second
                in self._keys_to_combine
            ]

    
    @overrides
    def text_to_instance(
        self, example: Dict[str, Any], features: Dict[str, Any], **metadata
    ) -> Instance:
        # from datasets.features import Value, ClassLabel

        # features = features.copy()

        fields: Dict[str, Field] = {}
        # return Instance(fields)

        metadata["_example"] = example
        metadata["_features"] = features

        raw: Dict[str, Any] = {}
        metadata["_raw"] = raw

        for key, value in metadata.items():
            fields[key] = MetadataField(value)

        for key, feature in features.items():
            value = example.get(key)
            field: Optional[Field]
            if 'label' in key:
                # print(value)
                assert isinstance(value, int)
                if value >= 0:
                    # label = feature.int2str(value)
                    field = LabelField(
                        label=self.class_mappings[self.scenario][value],
                        label_id=value,
                        label_namespace=f"{key}s",
                        skip_indexing=True,
                    )
                    raw[key] = value
                    # label_namespace = f'{key}s'
                    # field = LabelField(label, label_namespace)
                else:
                    field = None
            # TODO smooth labels
            elif isinstance(feature, SentencePairFeature):
                if feature.key1 in example and feature.key2 in example:
                    field = self._string_pair_to_field(example[feature.key1], example[feature.key2])
                else:
                    field = self._string_value_to_field(value)
                raw[feature.key1] = example.get(f"_raw_{feature.key1}", example.get(feature.key1))
                raw[feature.key2] = example.get(f"_raw_{feature.key2}", example.get(feature.key1))

            elif isinstance(feature, Value) and pa.types.is_string(feature.pa_type):
                field = self._string_value_to_field(value)
                raw[key] = value
            # elif isinstance(feature, Value) and pa.types.is_boolean(feature.pa_type):
            #     ...
            # elif isinstance(feature, Value) and pa.types.is_integer(feature.pa_type):
            #     ...
            # elif isinstance(feature, Value) and pa.types.is_floating(feature.pa_type):
            #     ...
            elif isinstance(feature, Value):
                if pa.types.is_floating(feature.pa_type):
                    field = ArrayField(np.array(value, dtype=feature.dtype))
                else:
                    field = MetadataField(value)
                raw[key] = value
            elif isinstance(feature, Sequence):
                field = ArrayField(np.array(value, dtype=feature.feature.dtype))
            else:
                raise NotImplementedError

            # Some fields may be absent in test time
            if field is not None:
                fields[key] = field

        instance = Instance(fields)
        return instance
        # if pickled:
        #     return pickle.dumps(instance)

        # if self._combine_input_fields:
        #     tokens = self._tokenizer.add_special_tokens(premise, hypothesis)
        #     fields["tokens"] = TextField(tokens, self._token_indexers)
        # else:
        #     premise_tokens = self._tokenizer.add_special_tokens(premise)
        #     hypothesis_tokens = self._tokenizer.add_special_tokens(hypothesis)
        #     fields["premise"] = TextField(premise_tokens, self._token_indexers)
        #     fields["hypothesis"] = TextField(hypothesis_tokens, self._token_indexers)

        #     metadata = {
        #         "premise_tokens": [x.text for x in premise_tokens],
        #         "hypothesis_tokens": [x.text for x in hypothesis_tokens],
        #     }
        #     fields["metadata"] = MetadataField(metadata)

        # if label:
        #     fields["label"] = LabelField(label)

        # for key, value in metadata.items():
        #     assert key not in fields
        #     fields[key] = MetadataField(value)

        # return Instance(fields)


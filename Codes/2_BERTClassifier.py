import keras
import tensorflow as tf
from numpy.random import seed
import random
from tensorflow.random import set_seed
import os
from transformers import *
from transformers.modeling_tf_outputs import *
from transformers.modeling_tf_utils import *
def initialize():
    seed(1)
    random.seed(1)
    set_seed(1)
    os.environ['PYTHONHASHSEED'] = '1'
def get_initializer(initializer_range: float = 0.02, seed=1) -> tf.initializers.TruncatedNormal:
    """
    Creates a :obj:`tf.initializers.TruncatedNormal` with the given range.

    Args:
        initializer_range (`float`, defaults to 0.02): Standard deviation of the initializer range.

    Returns:
        :obj:`tf.initializers.TruncatedNormal`: The truncated normal initializer.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range, seed = seed)

class TFRobertaClassificationHead(tf.keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, **kwargs):
        super(TFRobertaClassificationHead, self).__init__(config, **kwargs)
        self.dropout = tf.keras.layers.Dropout(.25)
        self.dense = tf.keras.layers.Dense(128,
                                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1),bias_initializer = tf.keras.initializers.GlorotUniform(seed=1),
                                              activation='tanh',
                                              name="classifier-1")
        self.out_proj = tf.keras.layers.Dense(config.num_labels,
                                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1),bias_initializer = tf.keras.initializers.GlorotUniform(seed=1),
                                              name="out_proj")

    def call(self, features, training=False):
#         x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(features)
        x = self.dense(x)
        x = self.dropout(x)
#         x = self.dense_2(x)
#         x = self.dropout_2(x, training = training)
        x = self.out_proj(x)
        return x


class TFRobertaForSequenceClassificationTemp(TFRobertaPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **logits**: ``tf.Tensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

        tokenizer = RoertaTokenizer.from_pretrained('roberta-base')
        model = TFRobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        labels = tf.constant([1])[None, :]  # Batch size 1
        outputs = model(input_ids)
        logits = outputs[0]

    """
    def __init__(self, config, *inputs, **kwargs):
        super(TFRobertaForSequenceClassificationTemp, self).__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.roberta = TFRobertaMainLayer(config, name="roberta")
        self.classifier = TFRobertaClassificationHead(config, name="classifier")
    
    def call(self, inputs, **kwargs):
        outputs = self.roberta(inputs, **kwargs)
#         print(outputs)
        sequence_output = outputs.pooler_output
        logits = self.classifier(sequence_output, training=kwargs.get('training', False))

        outputs = (logits,) + outputs[2:]

        return outputs  # logits, (hidden_states), (attentions)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class TFXLNetForSequenceClassificationTemp(TFXLNetPreTrainedModel, TFSequenceClassificationLoss):
        def __init__(self, config, *inputs, **kwargs):
            super().__init__(config, *inputs, **kwargs)
            self.num_labels = config.num_labels
            self.dropout = tf.keras.layers.Dropout(.25)
            self.transformer = TFXLNetMainLayer(config, name="transformer")
            self.sequence_summary = TFSequenceSummary(
                config, initializer_range=config.initializer_range, name="sequence_summary"
            )
            self.classifier = tf.keras.layers.Dense(
                128, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1),bias_initializer = tf.keras.initializers.GlorotUniform(seed=1), name="classifier", activation = 'tanh'
            )
            self.logits_proj = tf.keras.layers.Dense(
                config.num_labels, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1),bias_initializer = tf.keras.initializers.GlorotUniform(seed=1), name="logits_proj"
            )


        def call(
            self,
            input_ids=None,
            attention_mask=None,
            mems=None,
            perm_mask=None,
            target_mapping=None,
            token_type_ids=None,
            input_mask=None,
            head_mask=None,
            inputs_embeds=None,
            use_mems=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
            training=False,
            **kwargs,
        ):
            r"""
            labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for computing the sequence classification/regression loss. Indices should be in ``[0, ...,
                config.num_labels - 1]``. If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
                If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
            """
            inputs = input_processing(
                func=self.call,
                config=self.config,
                input_ids=input_ids,
                attention_mask=attention_mask,
                mems=mems,
                perm_mask=perm_mask,
                target_mapping=target_mapping,
                token_type_ids=token_type_ids,
                input_mask=input_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_mems=use_mems,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
                training=training,
                kwargs_call=kwargs,
            )
            transformer_outputs = self.transformer(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                mems=inputs["mems"],
                perm_mask=inputs["perm_mask"],
                target_mapping=inputs["target_mapping"],
                token_type_ids=inputs["token_type_ids"],
                input_mask=inputs["input_mask"],
                head_mask=inputs["head_mask"],
                inputs_embeds=inputs["inputs_embeds"],
                use_mems=inputs["use_mems"],
                output_attentions=inputs["output_attentions"],
                output_hidden_states=inputs["output_hidden_states"],
                return_dict=return_dict,
                training=inputs["training"],
            )
            output = transformer_outputs[0]
            output = self.sequence_summary(output)

#             output = self.sequence_summary(output)
            output = self.dropout(output)
            logits = self.classifier(output)
            logits = self.dropout(logits)
            
            logits = self.logits_proj(logits)

            outputs = (logits,) + transformer_outputs[2:]
            return outputs


        def serving_output(self, output):
            hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
            attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None
            mems = tf.convert_to_tensor(output.mems) if output.mems is not None else None

            return TFXLNetForSequenceClassificationOutput(
                logits=output.logits, mems=mems, hidden_states=hs, attentions=attns
            )

        
class TFBertForSequenceClassificationTemp(TFBertPreTrainedModel, TFSequenceClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"mlm___cls", r"nsp___cls", r"cls.predictions", r"cls.seq_relationship"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels

        self.bert = TFBertMainLayer(config, name="bert")
        self.dropout = tf.keras.layers.Dropout(.25)
        self.classifier_2 = tf.keras.layers.Dense(
            units=128,
             kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1),bias_initializer = tf.keras.initializers.GlorotUniform(seed=1), name="classifier", activation='tanh'
        )
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,
             kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1),bias_initializer = tf.keras.initializers.GlorotUniform(seed=1), name="output_layer"
        )

    def call(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        output_attentions= None,
        output_hidden_states = None,
        return_dict = None,
        labels = None,
        training= False,
        **kwargs,
    ):
        r"""
        labels (:obj:`tf.Tensor` or :obj:`np.ndarray` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(inputs=pooled_output)
        logits = self.classifier_2(inputs=pooled_output)
        logits = self.dropout(logits)
        logits = self.classifier(inputs=logits)
        return (logits,) + outputs[2:]
    
class TFDistilBertForSequenceClassificationTemp(TFDistilBertPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.distilbert = TFDistilBertMainLayer(config, name="distilbert")
        self.pre_classifier = tf.keras.layers.Dense(
            config.dim,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="relu",
            name="pre_classifier",
        )
        self.classifier_2 = tf.keras.layers.Dense(
            units=128,
             kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1),bias_initializer = tf.keras.initializers.GlorotUniform(seed=1), name="classifier", activation='tanh'
        )
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,
             kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1),bias_initializer = tf.keras.initializers.GlorotUniform(seed=1), name="output_layer"
        )
        self.dropout = tf.keras.layers.Dropout(.25)


    def call(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in ``[0, ...,
            config.num_labels - 1]``. If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )
        distilbert_output = self.distilbert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
#         print(distilbert_output)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
#         pooled_output = distilbert_output.pooler_output  # (bs, seq_len, dim)
#         pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.classifier_2(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
#         print(pooled_output)
        logits = self.classifier(pooled_output)
        
#         logits = self.classifier(logits)  # (bs, dim)
        return (logits,) + distilbert_output[2:]

        


    # Copied from transformers.models.bert.modeling_tf_bert.TFBertForSequenceClassification.serving_output
    def serving_output(self, output: TFSequenceClassifierOutput) -> TFSequenceClassifierOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFSequenceClassifierOutput(logits=output.logits, hidden_states=hs, attentions=attns)

# x = TFXLNetForSequenceClassificationTemp.from_pretrained('xlnet-base-cased')
# tkn = XLNetTokenizer.from_pretrained('xlnet-base-cased')
# c = tkn.encode('HI')
# print(x.predict([c]))




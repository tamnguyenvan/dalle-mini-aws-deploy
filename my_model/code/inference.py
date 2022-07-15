import os
import json
import random
from functools import partial

import numpy as np

from PIL import Image

from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from dalle_mini import DalleBartProcessor

import jax
import jax.numpy as jnp
from flax.jax_utils import replicate

from flax.training.common_utils import shard_prng_key
from PIL import Image


def model_fn(model_dir):
    dalle_dir = os.path.join(model_dir, 'dalle')
    vqgan_dir = os.path.join(model_dir, 'vqgan')

    dalle_model, params = DalleBart.from_pretrained(
        dalle_dir, dtype=jnp.float16, _do_init=False
    )
    vqgan, vqgan_params = VQModel.from_pretrained(
        vqgan_dir, _do_init=False
    )
    processor = DalleBartProcessor.from_pretrained(dalle_dir)
    return dalle_model, params, vqgan, vqgan_params, processor


def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    if request_content_type == 'application/json':
        prompt_data = json.loads(request_body)
        if not 'prompt' in prompt_data:
            raise Exception('Text prompt must be provided')
        return prompt_data
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        raise Exception(f'Unsupported content type: {request_content_type}')


def predict_fn(input_data, model):
    dalle_model, params, vqgan, vqgan_params, processor = model
    params = replicate(params)
    vqgan_params = replicate(vqgan_params)

    # model inference
    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
    def p_generate(
        tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
    ):
        return dalle_model.generate(
            **tokenized_prompt,
            prng_key=key,
            params=params,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            condition_scale=condition_scale,
        )

    # decode image
    @partial(jax.pmap, axis_name="batch")
    def p_decode(indices, vqgan_params):
        return vqgan.decode_code(indices, params=vqgan_params)


    prompts = [
        input_data['prompt']
    ]
    tokenized_prompts = processor(prompts)
    tokenized_prompt = replicate(tokenized_prompts)

    # number of predictions per prompt
    n_predictions = input_data.get('n_predictions', 1)

    # We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
    gen_top_k = None
    gen_top_p = None
    temperature = None
    cond_scale = 10.0

    # create a random key
    seed = random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)

    # generate images
    images = []
    for i in range(max(n_predictions // jax.device_count(), 1)):
        # get a new key
        key, subkey = jax.random.split(key)
        # generate images
        encoded_images = p_generate(
            tokenized_prompt,
            shard_prng_key(subkey),
            params,
            gen_top_k,
            gen_top_p,
            temperature,
            cond_scale,
        )
        # remove BOS
        encoded_images = encoded_images.sequences[..., 1:]
        # decode images
        decoded_images = p_decode(encoded_images, vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        for decoded_img in decoded_images:
            img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
            images.append(img)
    return images[0]


def output_fn(prediction, accept):
    assert accept == 'application/json'
    resp = {
        'image': np.array(prediction).tolist()
    }
    return json.dumps(resp)
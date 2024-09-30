from diffusers import KolorsPipeline
from diffusers.pipelines.kolors.pipeline_output import KolorsPipelineOutput
from torchvision.transforms.v2 import PILToTensor
from pathlib import Path
import xformers
import torch
import os


class KolorsWrapper:
    """
  A wrapper class around the Hugging Face KolorsPipeline that simplifies prompt conditioning, seed management, and
  image generation via the Kolors diffusion model.

  Attributes:
      main_pipeline (KolorsPipeline): The Kolors diffusion model pipeline.
  """

    def __init__(self, checkpoint: str | Path | None = None):

        """
    Initialize the KolorsWrapper with a checkpoint or download the default pre-trained model.

    Args:
        checkpoint (str | Path | None): Path to the saved model checkpoint. If None, downloads the pre-trained Kolors model.
    """

        if isinstance(checkpoint, str):
            checkpoint = Path(checkpoint)
        if checkpoint is None or not checkpoint.exists():
            self.main_pipeline = KolorsPipeline.from_pretrained("Kwai-Kolors/Kolors-diffusers",
                                                                torch_dtype=torch.float16, variant="fp16")
            self.main_pipeline.save_pretrained("kolors_diffusers")
        else:
            self.main_pipeline = KolorsPipeline.from_pretrained(checkpoint, torch_dtype=torch.float16, variant="fp16")
        if not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA device is not available. The KolorsWrapper requires a GPU with CUDA support for optimal '
                'performance.')
        self.main_pipeline.to('cuda')
        self.main_pipeline.set_progress_bar_config(leave=False, disable=True)
        self.main_pipeline.enable_xformers_memory_efficient_attention()

    def _sanitize_parameters(self, **kwargs) -> tuple[dict, dict, dict]:

        """
    Sanitize and organize input parameters into groups for preprocessing, forward pass, and postprocessing.

    Args:
        **kwargs: Keyword arguments for conditioning, seed, and other model parameters.

    Returns:
        tuple: Three dictionaries containing parameters for pre-processing, forward pass, and post-processing respectively.
    """

        pre_params = {}
        forw_params = {}
        post_params = {}
        if 'seed' not in kwargs or not isinstance(kwargs['seed'], int):
            forw_params['seed'] = self.__get_random_seed()
        else:
            forw_params['seed'] = kwargs['seed']
        post_params['seed'] = forw_params['seed']
        if 'return_pt' in kwargs and isinstance(kwargs['return_pt'], bool):
            post_params['return_pt'] = kwargs['return_pt']
        else:
            post_params['return_pt'] = False
        if 'num_inference_steps' not in kwargs or not isinstance(kwargs['num_inference_steps'], int):
            forw_params['num_inference_steps'] = 25
        else:
            forw_params['num_inference_steps'] = kwargs['num_inference_steps']
        if 'conditioning' in kwargs and isinstance(kwargs['conditioning'], str):
            pre_params['conditioning'] = kwargs['conditioning'].strip()
        else:
            pre_params['conditioning'] = ''
        if 'images_per_prompt' in kwargs:
            if isinstance(kwargs['images_per_prompt'], int) and kwargs['images_per_prompt'] > 0:
                pre_params['images_per_prompt'] = kwargs['images_per_prompt']
            else:
                pre_params['images_per_prompt'] = 1
        else:
            pre_params['images_per_prompt'] = 1
        if 'guidance_scale' in kwargs and isinstance(kwargs['guidance_scale'], (int, float)):
            if kwargs['guidance_scale'] >= 1.0:
                forw_params['guidance_scale'] = kwargs['guidance_scale']
            else:
                forw_params['guidance_scale'] = 5.0
        else:
            forw_params['guidance_scale'] = 5.0
        post_params['guidance_scale'] = forw_params['guidance_scale']
        return pre_params, forw_params, post_params

    def preprocess(self, inputs: str | list[str], **kwargs) -> list[str]:

        """
    Preprocess the input prompts by applying conditioning and formatting them for model input.

    Args:
        inputs (str | list[str]): The prompt(s) used for image generation.
        **kwargs: Additional keyword arguments like conditioning and images_per_prompt.

    Returns:
        list[str]: A list of prompts formatted for the diffusion model.

    Raises:
        ValueError: If the input type is not a string or list of strings.
    """

        if not isinstance(inputs, (str, list)):
            raise ValueError(
                f'{type(inputs)} is not a valid prompt input type. It must be either a string or a list of strings.')

        if isinstance(inputs, str):
            prompts = [(kwargs['conditioning'] + ' ' + inputs).strip()] * kwargs['images_per_prompt']

        if isinstance(inputs, list):
            prompts = []
            for p in inputs:
                prompts.extend([(kwargs['conditioning'] + ' ' + p).strip()] * kwargs['images_per_prompt'])
        return prompts

    def _forward(self, inputs: list[str], **kwargs) -> KolorsPipelineOutput:

        """
    Perform the forward pass through the diffusion model to generate images.

    Args:
        inputs (list[str]): Preprocessed prompts.
        **kwargs: Forward parameters like seed and number of inference steps.

    Returns:
        KolorsPipelineOutput: The output of the KolorsPipeline containing generated images.
    """

        generator = torch.Generator('cuda').manual_seed(kwargs['seed'])
        model_out = self.main_pipeline(inputs, generator=generator, num_inference_steps=kwargs['num_inference_steps'],
                                       guidance_scale=kwargs['guidance_scale'])
        return model_out

    def postprocess(self, output: KolorsPipelineOutput, **kwargs) -> dict:

        """
    Post-process the model output, optionally converting images to PyTorch tensors.

    Args:
        output (KolorsPipelineOutput): The raw output from the diffusion model.
        **kwargs: Post-processing options such as returning images as tensors.

    Returns:
        dict: A dictionary containing the seed and generated images (optionally as PyTorch tensors).
    """

        pipeline_output = {}
        pipeline_output['seed'] = kwargs['seed']
        to_tensor = PILToTensor()
        if kwargs['return_pt']:
            convert = lambda x: to_tensor(x)
        else:
            convert = lambda x: x
        pipeline_output['images'] = [convert(img) for img in output.images]
        pipeline_output['guidance_scale'] = kwargs['guidance_scale']
        return pipeline_output

    def __call__(self, inputs: str | list[str], **kwargs):

        """
    End-to-end pipeline for generating images from prompts. Handles preprocessing, forward pass, and postprocessing.

    Args:
        inputs (str | list[str]): The input prompts for image generation.
        **kwargs: Optional parameters for preprocessing, forward pass, and postprocessing.

    Returns:
        dict: The processed output containing generated images and seed information.
    """

        pre, forw, post = self._sanitize_parameters(**kwargs)
        model_input = self.preprocess(inputs, **pre)
        outputs = self._forward(model_input, **forw)
        post_output = self.postprocess(outputs, **post)
        return post_output

    def __get_random_seed(self) -> int:

        """
    Generate a random seed using os.urandom.

    Returns:
        int: A random integer seed for reproducibility in image generation.
    """

        RAND_SIZE = 8
        random_data = os.urandom(RAND_SIZE)
        random_seed = int.from_bytes(random_data, byteorder="big")
        return random_seed
